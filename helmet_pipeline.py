import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from local_runtime import configure

configure()
from ultralytics import YOLO


BBox = List[float]
DetectionResult = Dict[str, Any]


def apply_vlm_review(result, reviewer, image, stress_image=None):
    if reviewer is None or result['decision'] != 'uncertain':
        return result
    use_stress = stress_image is not None and stress_image.size > 0
    decision, status = reviewer.review(stress_image if use_stress else image)
    return {
        **result,
        'small_model_decision': 'uncertain',
        'decision': 'no_violation' if decision == 'ok' else decision,
        'decision_source': 'large_model',
        'has_violation': None if decision == 'uncertain' else decision == 'violation',
        'need_manual_review': decision == 'uncertain',
        'review_status': status,
        'review_view': 'stress' if use_stress else 'original',
    }


@dataclass(frozen=True)
class AlgRule:
    alg_code: str
    positive_classes: Tuple[int, ...]
    related_classes: Tuple[int, ...]
    require_scene_gate: bool = False
    require_ppxw_anchor: bool = False
    stress_profile: str = 'robust'
    description: str = ''



ALG_RULES = {
    '00001001': AlgRule('00001001', (1,), (0, 1), False, False, 'appearance', '未戴安全帽'),
}



RELATED_PRIORITY = {
    '00001001': (1, 0),
}


class AnJianDetection:
    def __init__(
        self,
        save_aux_images: bool = True,
        aux_image_dir: str = './saved_multi_view_images',
        gpu_device: str = '1',
        debug_logging: bool = True,
        model_path: Optional[str] = None,
    ):
        if model_path is None:
            raise ValueError('Provide a locally trained checkpoint with --model.')
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.jiance_model = YOLO(str(self.model_path.resolve()))
        self.cj_model = None
        self.gpu_device = str(gpu_device)
        self.debug_logging = bool(debug_logging)
        self.save_aux_images = save_aux_images
        self.aux_image_dir = Path(aux_image_dir)
        if self.save_aux_images:
            self.aux_image_dir.mkdir(parents=True, exist_ok=True)

        print(f'[AnJianDetection] 当前小模型推理显卡: cuda:{self.gpu_device}')
        print(f'[AnJianDetection] 详细日志开关: {self.debug_logging}')
        print(f'[AnJianDetection] 小模型权重: {self.model_path.resolve()}')

        # ==================== 超参数调整区 ====================
        self.base_conf = float(os.getenv('AJ_BASE_CONF', '0.25'))
        self.ppxw_conf = float(os.getenv('AJ_PPXW_CONF', '0.5'))
        self.focus_crop_scale = float(os.getenv('AJ_FOCUS_CROP_SCALE', '1.5'))
        self.focus_support_iou = float(os.getenv('AJ_FOCUS_SUPPORT_IOU', '0.18'))

        # 双阈值
        self.direct_positive_score = float(os.getenv('AJ_DIRECT_POSITIVE_SCORE', '0.75')) # 高阈值：直接采信
        self.low_positive_score = float(os.getenv('AJ_LOW_POSITIVE_SCORE', '0.50'))       # 低阈值：送入复核

        self.direct_multi_positive_score = float(os.getenv('AJ_DIRECT_MULTI_POSITIVE_SCORE', '0.70'))
        self.direct_multi_positive_count = int(os.getenv('AJ_DIRECT_MULTI_POSITIVE_COUNT', '2'))

        self.risk_threshold = float(os.getenv('AJ_RISK_THRESHOLD', '0.65'))
        self.related_missing_penalty = float(os.getenv('AJ_RELATED_MISSING_PENALTY', '0.65'))
        self.view_swap_sigma = float(os.getenv('AJ_VIEW_SWAP_SIGMA', '25'))
        self.view2_variant = os.getenv('AJ_VIEW2_VARIANT', 'gaussian').strip().lower() or 'gaussian'
        self.ablation_mode = os.getenv('AJ_ABLATION_MODE', 'full').strip().lower() or 'full'
        self.strong_semantic_algcodes = {'00001002', '01010102', '06001006', '06001008', '91001001', '91003006'}

    def _debug(self, message: str):
        if self.debug_logging:
            print(message)

    def _format_score(self, score: float) -> str:
        return f'{float(score):.4f}'

    def _describe_result(self, result: DetectionResult) -> str:
        scores = result.get('scores', []) or []
        categories = result.get('categories', []) or []
        if not scores:
            return 'count=0, categories=[], top_score=None'
        top_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        top_cat = int(categories[top_idx]) if top_idx < len(categories) else None
        top_score = self._format_score(scores[top_idx])
        category_counter: Dict[int, int] = {}
        for cls in categories:
            key = int(cls)
            category_counter[key] = category_counter.get(key, 0) + 1
        category_text = ', '.join(f'{k}:{v}' for k, v in sorted(category_counter.items()))
        return f'count={len(scores)}, categories=[{category_text}], top_category={top_cat}, top_score={top_score}'

    def _log_scene_result(self, view_tag: str, alg_code: str, scene_classes: List[str], condition_met: bool):
        self._debug(f'[scene_model][{view_tag}][alg={alg_code}] scene_classes={scene_classes}, condition_met={condition_met}')

    def _log_detect_result(self, view_tag: str, alg_code: str, classes: Tuple[int, ...], conf: float, result: DetectionResult):
        self._debug(f'[detect_model][{view_tag}][alg={alg_code}] classes={list(classes)}, conf={conf}, {self._describe_result(result)}')

    def _log_view_summary(self, alg_code: str, tag: str, summary: Dict[str, Any]):
        self._debug(f'[view_summary][{tag}][alg={alg_code}] {summary}')

    def _log_risk(self, alg_code: str, risk_info: Dict[str, Any]):
        self._debug(f'[risk][alg={alg_code}] {risk_info}')

    def _log_final_decision(self, result: Dict[str, Any]):
        self._debug(
            f"[final_decision] decision={result.get('decision')}, has_violation={result.get('has_violation')}, "
            f"decision_source={result.get('decision_source')}, "
            f"need_manual_review={result.get('need_manual_review')}"
        )

    def get_rule(self, alg_code: str) -> AlgRule:
        if alg_code not in ALG_RULES:
            raise KeyError(f'未配置的违章编码: {alg_code}')
        return ALG_RULES[alg_code]

    def model_jiance(self, image, classes, conf=0.5, imgsz=1280):
        imgsz = min(int(imgsz), 640)
        return self.jiance_model(image, classes=list(classes), conf=conf, imgsz=imgsz, half=self.gpu_device != 'cpu', device=self.gpu_device, verbose=self.debug_logging)

    def model_class(self, image, classes=(0, 1, 2, 3, 4), conf=0.1, imgsz=1280):
        if self.cj_model is None:
            return []
        result = self.cj_model(image, classes=list(classes), conf=conf, imgsz=imgsz, half=True, device=self.gpu_device)[0]
        return [str(int(cls)) for cls in result.boxes.cls.tolist()]


    def _raw_detect(self, image, classes, conf=None, view_tag: str = 'unknown', alg_code: str = 'unknown') -> DetectionResult:
        conf = self.base_conf if conf is None else conf
        result = self.model_jiance(image, classes=classes, conf=conf, imgsz=1280)[0]
        parsed = {
            'coords': result.boxes.xyxy[:].tolist(),
            'categories': result.boxes.cls[:].tolist(),
            'scores': result.boxes.conf[:].tolist(),
        }
        self._log_detect_result(view_tag=view_tag, alg_code=alg_code, classes=classes, conf=conf, result=parsed)
        return parsed

    def _get_scene_condition(self, alg_code: str, image, view_tag: str = 'unknown') -> Tuple[bool, List[str]]:
        rule = self.get_rule(alg_code)
        if not rule.require_scene_gate:
            scene_classes: List[str] = []
            condition_met = True
            self._log_scene_result(view_tag, alg_code, scene_classes, condition_met)
            return condition_met, scene_classes
        scene_classes = self.model_class(image)
        flag = '0' not in scene_classes and '2' in scene_classes
        if alg_code in ['01010102', '91003006']:
            condition_met = (all(item in scene_classes for item in ['1', '2']) or flag)
        else:
            condition_met = True
        self._log_scene_result(view_tag, alg_code, scene_classes, condition_met)
        return condition_met, scene_classes

    def iou_pt(self, box_list, box2):
        x2min, y2min, x2max, y2max = box2
        for box1 in box_list or []:
            x1min, y1min, x1max, y1max = box1
            if x1max < x2min or x2max < x1min or y1max < y2min or y2max < y1min:
                continue
            s1 = (y1max - y1min + 1.0) * (x1max - x1min + 1.0)
            s2 = (y2max - y2min + 1.0) * (x2max - x2min + 1.0)
            xmin = max(x1min, x2min)
            ymin = max(y1min, y2min)
            xmax = min(x1max, x2max)
            ymax = min(y1max, y2max)
            intersection = max(ymax - ymin + 1, 0) * max(xmax - xmin + 1, 0)
            union = s1 + s2 - intersection
            if union > 0 and intersection / union >= 0.2:
                return 1
        return 0

    def _filter_by_iou(self, anchor_boxes, result: DetectionResult) -> DetectionResult:
        filtered_indices = [idx for idx, box in enumerate(result['coords']) if self.iou_pt(anchor_boxes, box)]
        return {
            'coords': [result['coords'][i] for i in filtered_indices],
            'categories': [result['categories'][i] for i in filtered_indices],
            'scores': [result['scores'][i] for i in filtered_indices],
        }

    def _run_group_detection(self, alg_code: str, image, classes: Tuple[int, ...], view_tag: str = 'unknown', scene_condition_cache: Optional[Tuple[bool, List[str]]] = None) -> DetectionResult:
        rule = self.get_rule(alg_code)
        if scene_condition_cache is None:
            condition_met, scene_classes = self._get_scene_condition(alg_code, image, view_tag=view_tag)
        else:
            condition_met, scene_classes = scene_condition_cache
            self._debug(f'[scene_model][{view_tag}][alg={alg_code}] reused_from=base_image scene_classes={scene_classes}, condition_met={condition_met}')
        if rule.require_scene_gate and not condition_met:
            return {'coords': [], 'categories': [], 'scores': [], 'condition_met': False, 'scene_classes': scene_classes, 'reason': 'scene_condition_not_met'}

        result = self._raw_detect(image, classes=classes, conf=self.base_conf, view_tag=view_tag, alg_code=alg_code)
        if rule.require_ppxw_anchor:
            anchor_result = self._raw_detect(image, classes=(15,), conf=self.ppxw_conf, view_tag=f'{view_tag}|anchor', alg_code=alg_code)
            anchor_boxes = anchor_result.get('coords', [])
            if not anchor_boxes:
                return {'coords': [], 'categories': [], 'scores': [], 'condition_met': True, 'scene_classes': scene_classes, 'reason': 'no_high_work_bbox'}
            result = self._filter_by_iou(anchor_boxes, result)

        result.update({'condition_met': True, 'scene_classes': scene_classes, 'reason': 'ok'})
        self._debug(f'[run_group_detection][{view_tag}][alg={alg_code}] reason=ok, {self._describe_result(result)}')
        return result

    def _split_group_result(self, result: DetectionResult, positive_classes: Tuple[int, ...]) -> Dict[str, Any]:
        positive_set = set(positive_classes)
        pos_idx = [idx for idx, cls in enumerate(result.get('categories', [])) if int(cls) in positive_set]
        neg_idx = [idx for idx, cls in enumerate(result.get('categories', [])) if int(cls) not in positive_set]
        return {
            'all': result,
            'positive': {
                'coords': [result['coords'][i] for i in pos_idx],
                'categories': [result['categories'][i] for i in pos_idx],
                'scores': [result['scores'][i] for i in pos_idx],
            },
            'negative': {
                'coords': [result['coords'][i] for i in neg_idx],
                'categories': [result['categories'][i] for i in neg_idx],
                'scores': [result['scores'][i] for i in neg_idx],
            },
            'has_any': bool(result.get('scores')),
            'has_positive': bool(pos_idx),
            'has_negative': bool(neg_idx),
        }

    def _unsharp_mask(self, image, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    def _adjust_gamma(self, image, gamma: float) -> np.ndarray:
        gamma = max(float(gamma), 1e-6)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image, table)

    def _build_lowlight_stress_view(self, image, profile: str = 'robust'):
        profile = (profile or 'robust').lower()

        # Lift dark regions first, then denoise and enhance evidence so the
        # second view complements low-light scenes instead of repeating them.
        brightened = self._adjust_gamma(image, gamma=0.72)
        denoised = cv2.fastNlMeansDenoisingColored(brightened, None, 5, 5, 7, 21)
        denoised = cv2.bilateralFilter(denoised, d=7, sigmaColor=65, sigmaSpace=65)

        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        if profile == 'structure':
            clahe = cv2.createCLAHE(clipLimit=3.8, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            merged = cv2.merge((l_channel, a_channel, b_channel))
            structure_view = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            structure_view = self._unsharp_mask(structure_view, sigma=1.0, amount=0.7)
            return cv2.convertScaleAbs(structure_view, alpha=1.06, beta=6)

        if profile == 'small_object':
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
            l_channel = clahe.apply(l_channel)
            merged = cv2.merge((l_channel, a_channel, b_channel))
            detail_view = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            detail_view = cv2.convertScaleAbs(detail_view, alpha=1.08, beta=8)
            return self._unsharp_mask(detail_view, sigma=0.9, amount=0.85)

        clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        appearance_view = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(appearance_view, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=1.10, beta=4)
        hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=1.16, beta=10)
        appearance_view = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return self._unsharp_mask(appearance_view, sigma=1.0, amount=0.6)

    # =======================================================
    # [大胆重构 1]：先中值滤波灭噪点，再双边滤波保边缘
    # =======================================================
    def build_stress_view(self, image, profile: str = 'robust'):
        profile = (profile or 'robust').lower()
        if self.view2_variant == 'lowlight':
            return self._build_lowlight_stress_view(image, profile)

        # 中值滤波斩断极端孤立噪点
        denoised = cv2.medianBlur(image, 3)
        # 双边滤波保边平滑
        denoised = cv2.bilateralFilter(denoised, d=7, sigmaColor=75, sigmaSpace=75)

        if profile == 'appearance':
            hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=1.1)
            hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=1.05)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if profile == 'structure':
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        if profile == 'small_object':
            return self._unsharp_mask(denoised, sigma=1.5, amount=1.2)

        return denoised

    def build_focus_stress_view(self, image, profile: str = 'robust'):
        return self.build_stress_view(image, profile)

    def _build_related_highlight_view(self, image: np.ndarray, boxes: List[BBox]) -> Optional[np.ndarray]:
        if image is None or getattr(image, 'size', 0) == 0 or not boxes:
            return None
        h, w = image.shape[:2]
        base = image.copy()
        overlay = cv2.convertScaleAbs(base, alpha=0.82, beta=0)
        for box in boxes:
            expanded = self.expand_bbox(box, 1.2, w, h)
            x1, y1, x2, y2 = [int(v) for v in expanded]
            if x2 <= x1 or y2 <= y1:
                continue
            overlay[y1:y2, x1:x2] = base[y1:y2, x1:x2]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 215, 255), 1)
        return overlay

    def expand_bbox(self, bbox: BBox, scale: float, img_w: int, img_h: int) -> BBox:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        new_w = bw * scale
        new_h = bh * scale
        return [
            max(0, int(round(cx - new_w / 2.0))),
            max(0, int(round(cy - new_h / 2.0))),
            min(img_w - 1, int(round(cx + new_w / 2.0))),
            min(img_h - 1, int(round(cy + new_h / 2.0))),
        ]

    def crop_with_context(self, image, bbox: BBox, scale=1.5):
        h, w = image.shape[:2]
        crop_region = self.expand_bbox(bbox, scale, w, h)
        x1, y1, x2, y2 = crop_region
        crop = image[y1:y2, x1:x2].copy()
        return crop, crop_region

    def merge_boxes(self, boxes: List[BBox]) -> Optional[BBox]:
        if not boxes:
            return None
        xs1 = [box[0] for box in boxes]
        ys1 = [box[1] for box in boxes]
        xs2 = [box[2] for box in boxes]
        ys2 = [box[3] for box in boxes]
        return [min(xs1), min(ys1), max(xs2), max(ys2)]

    def map_boxes_from_crop(self, crop_coords: List[BBox], crop_region: BBox) -> List[BBox]:
        x_offset, y_offset = crop_region[0], crop_region[1]
        return [[x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset] for x1, y1, x2, y2 in crop_coords]

    def _calc_iou(self, box1: BBox, box2: BBox) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _build_view_save_prefix(self, alg_code: str) -> Path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return self.aux_image_dir / f'{alg_code}_{timestamp}'

    def _save_auxiliary_views(self, alg_code: str, stress_image: Optional[np.ndarray] = None, focus_image: Optional[np.ndarray] = None) -> Dict[str, str]:
        saved_paths: Dict[str, str] = {}
        if not self.save_aux_images:
            return saved_paths
        self.aux_image_dir.mkdir(parents=True, exist_ok=True)
        file_prefix = self._build_view_save_prefix(alg_code)
        if stress_image is not None and getattr(stress_image, 'size', 0) > 0:
            path = file_prefix.with_name(file_prefix.name + '_stress.jpg')
            if cv2.imwrite(str(path), stress_image):
                saved_paths['stress_image_path'] = str(path.resolve())
        if focus_image is not None and getattr(focus_image, 'size', 0) > 0:
            path = file_prefix.with_name(file_prefix.name + '_focus.jpg')
            if cv2.imwrite(str(path), focus_image):
                saved_paths['focus_image_path'] = str(path.resolve())
        return saved_paths

    def _select_main_box(self, result: DetectionResult, preferred_classes: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
        scores = result.get('scores', []) or []
        coords = result.get('coords', []) or []
        categories = result.get('categories', []) or []
        if not scores:
            return None
        preferred = set(preferred_classes)
        candidates: List[Tuple[float, int]] = []
        for idx, score in enumerate(scores):
            cls = int(categories[idx])
            area = max(1.0, (coords[idx][2] - coords[idx][0]) * (coords[idx][3] - coords[idx][1]))
            pref_bonus = 0.12 if cls in preferred else 0.0
            area_bonus = min(0.08, area / 250000.0)
            candidates.append((float(score) + pref_bonus + area_bonus, idx))
        _, best_idx = max(candidates, key=lambda x: x[0])
        return {
            'bbox': coords[best_idx],
            'category': int(categories[best_idx]),
            'score': float(scores[best_idx]),
            'index': int(best_idx),
        }

    def _select_top_k_boxes(self, result: DetectionResult, preferred_classes: Tuple[int, ...], k: int = 3) -> List[Dict[str, Any]]:
        scores = result.get('scores', []) or []
        coords = result.get('coords', []) or []
        categories = result.get('categories', []) or []
        if not scores:
            return []
        preferred = set(preferred_classes)
        ranked: List[Tuple[float, int]] = []
        for idx, score in enumerate(scores):
            cls = int(categories[idx])
            area = max(1.0, (coords[idx][2] - coords[idx][0]) * (coords[idx][3] - coords[idx][1]))
            pref_bonus = 0.12 if cls in preferred else 0.0
            area_bonus = min(0.08, area / 250000.0)
            ranked.append((float(score) + pref_bonus + area_bonus, idx))
        ranked.sort(key=lambda x: x[0], reverse=True)
        selected: List[Dict[str, Any]] = []
        for _, idx in ranked[:max(1, k)]:
            selected.append({
                'bbox': coords[idx],
                'category': int(categories[idx]),
                'score': float(scores[idx]),
                'index': int(idx),
            })
        return selected

    def _best_match(self, reference_box: BBox, result: DetectionResult) -> Optional[Dict[str, Any]]:
        best_iou = 0.0
        best_idx = -1
        for idx, box in enumerate(result.get('coords', [])):
            iou = self._calc_iou(reference_box, box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx < 0:
            return None
        return {
            'bbox': result['coords'][best_idx],
            'category': int(result['categories'][best_idx]),
            'score': float(result['scores'][best_idx]),
            'iou': float(best_iou),
            'index': int(best_idx),
        }

    def _summarize_view(self, group: Dict[str, Any], reference: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        positive = group['positive']
        negative = group['negative']
        all_result = group['all']
        pos_scores = [float(v) for v in positive.get('scores', [])]
        neg_scores = [float(v) for v in negative.get('scores', [])]
        top_positive_score = max(pos_scores) if pos_scores else 0.0
        top_negative_score = max(neg_scores) if neg_scores else 0.0
        top_positive_gap = max(0.0, top_positive_score - top_negative_score)
        label = 'no_violation'
        if group['has_positive']:
            label = 'violation'
        elif not group['has_any']:
            label = 'no_evidence'
        summary = {
            'has_any': group['has_any'],
            'has_positive': group['has_positive'],
            'has_negative': group['has_negative'],
            'label': label,
            'top_positive_score': round(top_positive_score, 4),
            'top_negative_score': round(top_negative_score, 4),
            'top_positive_gap': round(top_positive_gap, 4),
            'positive_count': len(pos_scores),
            'negative_count': len(neg_scores),
            'all_count': len(all_result.get('scores', [])),
            'main_positive': self._select_main_box(positive, tuple(int(c) for c in positive.get('categories', []))) if group['has_positive'] else None,
            'main_any': self._select_main_box(all_result, tuple(int(c) for c in all_result.get('categories', []))) if group['has_any'] else None,
        }
        if reference is not None:
            match = self._best_match(reference['bbox'], all_result)
            summary['reference_match'] = match
        return summary


    def _build_focus_view(self, image: np.ndarray, base_group: Dict[str, Any], rule: AlgRule) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        if not base_group['has_positive']:
            if not base_group['has_any']:
                return None, None
            source_box = self._select_main_box(base_group['all'], RELATED_PRIORITY.get(rule.alg_code, rule.related_classes))
            if source_box is None:
                return None, None
            focus_image = self._build_related_highlight_view(image, base_group['all'].get('coords', []))
            if focus_image is None:
                return None, None
            return focus_image, {
                'focus_type': 'related_review',
                'source_box': source_box,
                'crop_region': None,
            }

        source_box = self._select_main_box(base_group['positive'], rule.positive_classes)
        if source_box is None:
            return None, None

        crop, crop_region = self.crop_with_context(image, source_box['bbox'], self.focus_crop_scale)
        if crop.size == 0:
            return None, None

        focus_image = self.build_focus_stress_view(crop, rule.stress_profile)
        return focus_image, {
            'focus_type': 'violation_focus',
            'source_box': source_box,
            'crop_region': crop_region,
        }

    # =======================================================
    # [大胆重构 2]：加入前置逻辑拦截 (Hard Logic Gates)
    # =======================================================
    def _compute_risk(
        self,
        alg_code: str,
        base_group: Dict[str, Any],
        stress_group: Dict[str, Any],
        base_summary: Dict[str, Any],
        stress_summary: Dict[str, Any],
        focus_group: Optional[Dict[str, Any]] = None,
        focus_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base_label = base_summary['label']
        stress_label = stress_summary['label']
        focus_available = bool(focus_group and focus_summary and focus_group.get('has_any'))
        focus_label = focus_summary['label'] if focus_available else 'no_evidence'
        focus_type = focus_summary.get('focus_type', 'none') if focus_summary else 'none'

        base_pos = float(base_summary['top_positive_score'])
        stress_pos = float(stress_summary['top_positive_score'])
        focus_pos = float(focus_summary['top_positive_score']) if focus_available else 0.0

        stress_agrees = stress_group['has_positive'] and stress_pos >= (self.low_positive_score - 0.15)
        focus_agrees = bool(
            focus_available and
            focus_type == 'violation_focus' and
            focus_group['has_positive'] and
            focus_pos >= (self.low_positive_score - 0.12)
        )
        focus_related_confirms_stress = bool(
            focus_available and
            focus_type == 'related_review' and
            focus_group['has_positive'] and
            focus_pos >= (self.low_positive_score - 0.10)
        )

        if base_group['has_positive'] and base_pos >= self.low_positive_score:
            if not stress_agrees and not focus_agrees:
                return {
                    'algorithm': 'three_view_focus_confirm_v1',
                    'algCode': alg_code,
                    'needs_review': True,
                    'suggested_decision': 'violation',
                    'route_reason': 'veto_ghost_hallucination',
                    'risk_threshold': round(self.risk_threshold, 4),
                    'risk_total': 1.0,
                    'R_cls': 1.0,
                    'R_score': 1.0,
                    'R_evidence': 1.0,
                    'R_miss': 0.0,
                    'details': {
                        'base_positive_score': round(base_pos, 4),
                        'stress_positive_score': round(stress_pos, 4),
                        'focus_positive_score': round(focus_pos, 4),
                    },
                }

        if (not base_group['has_positive'] or base_pos < self.low_positive_score):
            if stress_group['has_positive'] and stress_pos >= self.low_positive_score:
                if focus_available and focus_type == 'related_review' and not focus_related_confirms_stress:
                    return {
                        'algorithm': 'three_view_focus_confirm_v1',
                        'algCode': alg_code,
                        'needs_review': False,
                        'suggested_decision': 'no_violation',
                        'route_reason': 'stress_positive_rejected_by_focus_review',
                        'risk_threshold': round(self.risk_threshold, 4),
                        'risk_total': 0.0,
                        'R_cls': 0.0,
                        'R_score': 0.0,
                        'R_evidence': 0.0,
                        'R_miss': 0.0,
                        'details': {
                            'base_positive_score': round(base_pos, 4),
                            'stress_positive_score': round(stress_pos, 4),
                            'focus_positive_score': round(focus_pos, 4),
                            'focus_type': focus_type,
                        },
                    }
                return {
                    'algorithm': 'three_view_focus_confirm_v1',
                    'algCode': alg_code,
                    'needs_review': True,
                    'suggested_decision': 'no_violation',
                    'route_reason': 'rescue_hidden_signal',
                    'risk_threshold': round(self.risk_threshold, 4),
                    'risk_total': 1.0,
                    'R_cls': 1.0,
                    'R_score': 0.0,
                    'R_evidence': 0.0,
                    'R_miss': 1.0,
                    'details': {
                        'base_positive_score': round(base_pos, 4),
                        'stress_positive_score': round(stress_pos, 4),
                        'focus_positive_score': round(focus_pos, 4),
                        'focus_type': focus_type,
                    },
                }

        if base_group['has_positive'] and focus_agrees and max(base_pos, focus_pos) >= self.low_positive_score:
            return {
                'algorithm': 'three_view_focus_confirm_v1',
                'algCode': alg_code,
                'needs_review': False,
                'suggested_decision': 'violation',
                'route_reason': 'focus_positive_confirm',
                'risk_threshold': round(self.risk_threshold, 4),
                'risk_total': 0.0,
                'R_cls': 0.0,
                'R_score': 0.0,
                'R_evidence': 0.0,
                'R_miss': 0.0,
                'details': {
                    'base_positive_score': round(base_pos, 4),
                    'stress_positive_score': round(stress_pos, 4),
                    'focus_positive_score': round(focus_pos, 4),
                },
            }

        if base_group['has_positive'] and base_pos >= self.direct_positive_score:
            return {
                'algorithm': 'three_view_focus_confirm_v1',
                'algCode': alg_code,
                'needs_review': False,
                'suggested_decision': 'violation',
                'route_reason': 'dual_threshold_high_pass',
                'risk_threshold': round(self.risk_threshold, 4),
                'risk_total': 0.0,
                'R_cls': 0.0,
                'R_score': 0.0,
                'R_evidence': 0.0,
                'R_miss': 0.0,
                'details': {'base_positive_score': round(base_pos, 4)},
            }

        focus_risk_active = bool(focus_available and focus_type == 'violation_focus')

        cls_conflicts = 0
        if base_label != stress_label and stress_label != 'no_evidence':
            cls_conflicts += 1
        if focus_risk_active and base_label != focus_label and focus_label != 'no_evidence':
            cls_conflicts += 1
        r_cls = min(1.0, 0.4 * cls_conflicts)

        score_drop_stress = max(0.0, base_pos - stress_pos)
        score_drop_focus = max(0.0, base_pos - focus_pos) if focus_risk_active else 0.0
        score_drop_mean = score_drop_stress if not focus_risk_active else (score_drop_stress + score_drop_focus) / 2.0
        r_score = min(1.0, 0.20 * score_drop_stress + 0.12 * score_drop_mean + (0.12 * score_drop_focus if focus_risk_active else 0.0))

        r_evidence = 0.0
        reference = base_summary['main_positive'] or base_summary['main_any']
        if reference is not None:
            stress_match = stress_summary.get('reference_match')
            if stress_match is None or float(stress_match.get('iou', 0.0)) < (self.focus_support_iou * 0.5):
                r_evidence += 0.35
            if focus_risk_active:
                focus_match = focus_summary.get('reference_match')
                if focus_match is None or float(focus_match.get('iou', 0.0)) < self.focus_support_iou:
                    r_evidence += 0.3

        r_miss = 0.0
        if (not base_group['has_positive']) and stress_group['has_positive']:
            r_miss += 0.6

        alpha, beta, gamma, delta = 0.15, 0.20, 0.40, 0.25
        total_risk = alpha * r_cls + beta * r_score + gamma * r_evidence + delta * r_miss

        needs_review = False
        suggested_decision = 'violation' if base_group['has_positive'] else 'no_violation'
        route_reason = 'risk_below_threshold_weighted'

        if base_group['has_positive'] and base_pos < self.low_positive_score:
            needs_review = True
            route_reason = 'dual_threshold_low_uncertain'
        elif base_group['has_positive'] and focus_risk_active and not focus_agrees and base_pos < self.direct_positive_score:
            needs_review = True
            route_reason = 'focus_not_confirmed_uncertain'
        elif total_risk >= self.risk_threshold:
            needs_review = True
            route_reason = 'risk_above_threshold_weighted'

        return {
            'algorithm': 'three_view_focus_confirm_v1',
            'algCode': alg_code,
            'needs_review': needs_review,
            'route_reason': route_reason,
            'suggested_decision': suggested_decision,
            'risk_threshold': round(self.risk_threshold, 4),
            'risk_total': round(total_risk, 4),
            'R_cls': round(r_cls, 4),
            'R_score': round(r_score, 4),
            'R_evidence': round(r_evidence, 4),
            'R_miss': round(r_miss, 4),
            'details': {
                'base_label': base_label,
                'stress_label': stress_label,
                'focus_label': focus_label,
                'focus_type': focus_type,
                'score_drop_stress': round(score_drop_stress, 4),
                'score_drop_focus': round(score_drop_focus, 4),
                'score_drop_mean': round(score_drop_mean, 4),
                'base_positive_score': round(base_pos, 4),
                'stress_positive_score': round(stress_pos, 4),
                'focus_positive_score': round(focus_pos, 4),
            },
        }


    def detect(self, alg_code: str, image, clean_image=None, noise_sigma: Optional[float] = None,
               *, reviewer=None) -> Dict[str, Any]:
        rule = self.get_rule(alg_code)
        self._debug(f'[detect_start] alg_code={alg_code}, description={rule.description}')
        if clean_image is None:
            clean_image = image

        base_result = self._run_group_detection(alg_code, image, rule.related_classes, view_tag='base_image')
        base_scene_cache = (base_result.get('condition_met', True), base_result.get('scene_classes', []))
        if not base_result.get('condition_met', True):
            result_payload = {
                'decision': 'no_violation',
                'has_violation': False,
                'need_manual_review': False,
                'boxes': [],
                'scores': [],
                'decision_source': 'small_model',
                'uncertainty_info': {
                    'algorithm': 'three_view_focus_confirm_v1',
                    'algCode': alg_code,
                    'needs_review': False,
                    'route_reason': 'scene_condition_not_met',
                    'details': {'scene_classes': base_result.get('scene_classes', [])},
                },
                'saved_view_paths': {},
            }
            self._log_final_decision(result_payload)
            return result_payload

        ablation_mode = getattr(self, 'ablation_mode', 'full')
        if ablation_mode not in {'full', 'no_view2', 'no_view3', 'base_only'}:
            ablation_mode = 'full'

        raw_base_group = self._split_group_result(base_result, rule.positive_classes)
        base_reference = self._select_main_box(raw_base_group['positive'], rule.positive_classes) if raw_base_group['has_positive'] else self._select_main_box(raw_base_group['all'], RELATED_PRIORITY.get(alg_code, rule.related_classes))
        base_summary = self._summarize_view(raw_base_group, reference=base_reference)
        self._log_view_summary(alg_code, 'base', base_summary)
        self._debug(f'[ablation][alg={alg_code}] mode={ablation_mode}')

        if ablation_mode == 'base_only':
            risk_info = {
                'algorithm': 'three_view_focus_confirm_v1',
                'algCode': alg_code,
                'ablation_mode': ablation_mode,
                'needs_review': not (raw_base_group['has_positive'] and float(base_summary['top_positive_score']) >= self.low_positive_score),
                'route_reason': 'base_only_direct_positive' if (raw_base_group['has_positive'] and float(base_summary['top_positive_score']) >= self.low_positive_score) else 'base_only_uncertain',
                'suggested_decision': 'violation' if raw_base_group['has_positive'] else 'no_violation',
                'risk_threshold': round(self.risk_threshold, 4),
                'risk_total': 0.0,
                'R_cls': 0.0,
                'R_score': 0.0,
                'R_evidence': 0.0,
                'R_miss': 0.0,
                'details': {
                    'base_positive_score': round(float(base_summary['top_positive_score']), 4),
                    'base_label': base_summary['label'],
                },
            }
            if raw_base_group['has_positive'] and float(base_summary['top_positive_score']) >= self.low_positive_score:
                result_payload = {
                    'decision': 'violation',
                    'has_violation': True,
                    'need_manual_review': False,
                    'boxes': raw_base_group['positive']['coords'],
                    'scores': raw_base_group['positive']['scores'],
                    'decision_source': 'small_model',
                    'uncertainty_info': risk_info,
                    'saved_view_paths': {},
                }
                self._log_final_decision(result_payload)
                return result_payload

            result_payload = {
                'decision': 'uncertain',
                'has_violation': None,
                'need_manual_review': True,
                'boxes': [],
                'scores': [],
                'decision_source': 'small_model',
                'uncertainty_info': risk_info,
                'saved_view_paths': {},
            }
            result_payload = apply_vlm_review(result_payload, reviewer, image)
            self._log_final_decision(result_payload)
            return result_payload

        disable_view2 = ablation_mode == 'no_view2'
        disable_view3 = ablation_mode == 'no_view3'

        if disable_view2:
            stress_image = None
            stress_result = {
                'coords': [],
                'categories': [],
                'scores': [],
                'condition_met': True,
                'scene_classes': base_scene_cache[1],
                'reason': 'ablation_no_view2',
            }
        else:
            stress_image = self.build_stress_view(image, rule.stress_profile)
            stress_result = self._run_group_detection(alg_code, stress_image, rule.related_classes, view_tag='stress_image', scene_condition_cache=base_scene_cache)
        raw_stress_group = self._split_group_result(stress_result, rule.positive_classes)

        swap_primary_view = (not disable_view2) and noise_sigma is not None and float(noise_sigma) >= self.view_swap_sigma
        primary_image = stress_image if swap_primary_view else image
        base_group = raw_stress_group if swap_primary_view else raw_base_group
        stress_group = raw_base_group if swap_primary_view else raw_stress_group

        base_reference = self._select_main_box(base_group['positive'], rule.positive_classes) if base_group['has_positive'] else self._select_main_box(base_group['all'], RELATED_PRIORITY.get(alg_code, rule.related_classes))
        base_summary = self._summarize_view(base_group, reference=base_reference)
        stress_summary = self._summarize_view(stress_group, reference=base_reference)
        self._log_view_summary(alg_code, 'base' if not swap_primary_view else 'base_swapped_stress', base_summary)
        self._log_view_summary(alg_code, 'stress' if not swap_primary_view else 'stress_swapped_raw', stress_summary)
        self._debug(f'[view_swap][alg={alg_code}] noise_sigma={noise_sigma}, swap_primary_view={swap_primary_view}')

        focus_image, focus_meta = (None, None) if disable_view3 else self._build_focus_view(primary_image, base_group, rule)
        focus_group = None
        focus_summary = None
        if focus_image is not None and focus_meta is not None:
            focus_result = self._run_group_detection(alg_code, focus_image, rule.related_classes, view_tag='focus_image', scene_condition_cache=base_scene_cache)
            if focus_meta.get('focus_type') == 'violation_focus' and focus_meta.get('crop_region') is not None:
                mapped_coords = self.map_boxes_from_crop(focus_result.get('coords', []), focus_meta['crop_region'])
                focus_result = {
                    'coords': mapped_coords,
                    'categories': focus_result.get('categories', []),
                    'scores': focus_result.get('scores', []),
                    'condition_met': True,
                    'scene_classes': base_scene_cache[1],
                    'reason': 'ok',
                }
            focus_group = self._split_group_result(focus_result, rule.positive_classes)
            focus_summary = self._summarize_view(focus_group, reference=focus_meta['source_box'])
            focus_summary['focus_type'] = focus_meta.get('focus_type', 'none')
            self._log_view_summary(alg_code, 'focus', focus_summary)

        saved_view_paths = self._save_auxiliary_views(alg_code, stress_image=stress_image, focus_image=focus_image)
        risk_info = self._compute_risk(alg_code, base_group, stress_group, base_summary, stress_summary, focus_group=focus_group, focus_summary=focus_summary)
        self._log_risk(alg_code, risk_info)

        if risk_info['needs_review']:
            result_payload = {
                'decision': 'uncertain',
                'has_violation': None,
                'need_manual_review': True,
                'boxes': [],
                'scores': [],
                'decision_source': 'small_model',
                'uncertainty_info': risk_info,
                'saved_view_paths': saved_view_paths,
            }
            result_payload = apply_vlm_review(result_payload, reviewer, image, stress_image)
            self._log_final_decision(result_payload)
            return result_payload

        suggested = risk_info.get('suggested_decision', 'no_violation')
        if suggested == 'violation':
            result_payload = {
                'decision': 'violation',
                'has_violation': True,
                'need_manual_review': False,
                'boxes': base_group['positive']['coords'],
                'scores': base_group['positive']['scores'],
                'decision_source': 'small_model',
                'uncertainty_info': risk_info,
                'saved_view_paths': saved_view_paths,
            }
            self._log_final_decision(result_payload)
            return result_payload

        result_payload = {
            'decision': 'no_violation',
            'has_violation': False,
            'need_manual_review': False,
            'boxes': [],
            'scores': [],
            'decision_source': 'small_model',
            'uncertainty_info': risk_info,
            'saved_view_paths': saved_view_paths,
        }
        self._log_final_decision(result_payload)
        return result_payload


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {image_path}')
    return image


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run the three-view helmet pipeline with a local checkpoint.')
    parser.add_argument('--image', type=str, required=True, help='Path to an input image.')
    parser.add_argument('--alg-code', type=str, default='00001001', help='Violation algorithm code.')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--vlm-review', action='store_true', help='Review uncertain results with a configured VLM.')
    args = parser.parse_args()

    reviewer = None
    if args.vlm_review:
        from vlm_review import VisionReviewer

        try:
            reviewer = VisionReviewer.from_environment()
        except ValueError as exc:
            parser.error(str(exc))
    detector = AnJianDetection(gpu_device=args.device, debug_logging=args.debug, save_aux_images=False, model_path=args.model)
    image = load_image(args.image)
    result = detector.detect(args.alg_code, image, reviewer=reviewer)
    print(json.dumps(result, ensure_ascii=False, indent=2))

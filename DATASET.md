# 20250731-PPE2286

Source: [20250731-PPE2286, v1, by Ying2001](https://universe.roboflow.com/ying2001/20250731-ppe2286-m0ega), distributed through Roboflow Universe under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset remains under that license; the code license does not apply to the images and annotations.

The experiment merged the original train and validation sets, then split the 2,286 image/label pairs by image-level class combinations (seed 42, target ratio 80/10/10). The resulting train/valid/test sizes are 1,828/231/227. Fifty-two out-of-bounds annotation rows were clipped. The manifest records the exact split, file hashes, and the final contents of all changed label files, so reconstruction does not depend on a random shuffle or re-rounding coordinates.

YOLO classes: 0 Helmet, 1 NoHelmet, 2 NoVest, 3 Vest. At image level, any class 1 annotation takes priority over class 0. The test set contains 117 compliant and 62 violating images; 48 without helmet-related ground truth are excluded from the image-level scores.

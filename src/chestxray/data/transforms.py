"""Image transforms.

Augmentation choices are deliberately anatomy-aware:

* **No horizontal flip.** Flipping an X-ray left-to-right puts the heart on the
  wrong side and is invalid for findings like cardiomegaly, so it is omitted.
* Small rotations / shifts / scale only, matching how a real chest film varies
  in positioning.
* ImageNet normalisation so pretrained backbones receive the input distribution
  they were trained on (essential for pretrained transfer-learning backbones).
"""
from __future__ import annotations

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool, clahe: bool = False):
    """Return an albumentations pipeline (imported lazily so the data-prep and
    metrics layers stay importable without the heavy vision stack).

    Set ``clahe=True`` to prepend Contrast-Limited Adaptive Histogram
    Equalisation — an engineered image feature that enhances lung-field
    contrast (see ``features.apply_clahe``).
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    pre = [A.CLAHE(clip_limit=2.0, p=1.0)] if clahe else []
    if train:
        aug = pre + [
            A.Resize(image_size, image_size),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=7,
                border_mode=0, p=0.5,
            ),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        aug = pre + [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    return A.Compose(aug)

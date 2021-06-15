from typing import List
import cv2
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor


class Dataset(object):
    def __init__(
        self, image_paths, bboxes=None, label=None, augmentations=None, backend="pil"
    ) -> None:
        self.image_paths = image_paths
        self.bboxes = bboxes
        self.label = label
        self.backend = backend
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        target = self.target[idx]
        if self.backend == "pil":
            img = Image.open(self.image_paths[idx])
            img = np.array(img)
        else:
            if self.grayscale is False:
                img = cv2.imread(self.image_paths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)

        bboxes = None
        if self.bboxes is not None:
            bboxes = self.bbox[idx]

        if self.augmentations is not None:
            if self.bboxes is not None:
                augmented = self.augmentations(image=img, bboxes=bboxes)
                img = augmented["image"]
                bboxes = augmented["bboxes"]
            else:
                augmented = self.augmentations(image=img)
                img = augmented["image"]

        image_tensor = torch.tensor(img, dtype=torch.float32)
        if self.grayscale:
            image_tensor = image_tensor.unsqueeze(0)
        return {
            "image": image_tensor,
            "targets": torch.tensor(target, torch.uint8),
            "bboxes": torch.tensor(bboxes, dtype=torch.int64)
            if bboxes is not None
            else None,
        }


def make_dataset(
    image_paths: List[str],
    bboxes: List[list],
    label: List(str),
    augmentation,
    backend: str,
    batch_size=8,
    shuffle=False,
    collate_fn=None,
    num_workers=1,
    pin_memory=False,
):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dataset = Dataset(image_paths, bboxes, label, augmentation, backend)
    return torch.utils.data.DataLoader(dataset, **kwargs)


def augmentation(aug):
    pass


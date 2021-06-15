import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn import metrics

from kami import KaMi
from model.factory import fetch_model
from utils.metrics import EvaluateMetric


class Model(KaMi):
    def __init__(self):
        super().__init__(config='config.yaml')
        if self.model:
            self.model_ft = fetch_model(self.model, pretrained=self.pretrained, num_classes=self.num_classes)

    def monitor_metrics(self, outputs, targets):
        return {
            k: EvaluateMetric(k)(outputs, targets) for k in self.metrics
        }

    # def fetch_optimizer(self):
    #     opt = torch.optim.Adam(self.parameters(), lr=3e-2)
    #     return opt

    # def fetch_scheduler(self):
    #     sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #         self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
    #     )
    #     return schevaluate

    def forward(self, inputs, targets=None):
        batch_size, _, _, _ = inputs.shape

        outputs = self.model_ft(inputs)

        if targets is not None:
            loss = self.loss(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, None, None


if __name__ == "__main__":
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    data_dir = "data/hymenoptera_data"
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=4, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    for inp, target in dataloaders['val']:
        print(target)
        break
    kami = Model()
    kami.fit(dataloaders["train"], dataloaders["val"])
    # test_loader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=1)
    # for idx, val in enumerate(kami.predict(test_loader)):
    #     print(idx, val)

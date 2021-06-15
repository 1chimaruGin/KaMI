"""
All classsification and detection models are from Ross Wightman's github repo 

https://github.com/rwightman/efficientdet-pytorch

https://github.com/rwightman/pytorch-image-models
"""

import timm
import torch
import torch.nn as nn
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


def fetch_model(
    task, model_name, num_classes, pretrained=False, image_size=512, **kwargs
):
    model = timm.create_model(model_name=model_name, pretrained=pretrained, **kwargs)

    if task == "classification":
        if "efficientnet" in model_name:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif "cspres" in model_name:
            in_features = model.head.fc.in_features
            model.head.fc = nn.Linear(in_features, num_classes)
        elif (
            model_name == "vit_large_patch16_384"
            or model_name == "deit_base_patch16_224"
        ):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        elif "resnext" in model_name:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

    elif task == "detection":
        ckpt = kwargs.pop("ckpt", None)
        if ckpt:
            checkpoint = torch.load(ckpt)
            pretrained = False
        else:
            pretrained = True

        if "efficientdet" in model_name:
            config = get_efficientdet_config(model_name)
            net = EfficientDet(config, pretrained_backbone=pretrained)
            net.load_state_dict(checkpoint)
            config.num_classes = num_classes
            config.image_size = image_size
            net.class_net = HeadNet(
                config,
                num_outputs=config.num_classes,
                norm_kwargs=dict(eps=0.001, momentum=0.01),
            )
            return DetBenchTrain(net, config)()

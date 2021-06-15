import torch
import numpy as np
from sklearn import metrics

from utils import *

np.seterr(divide='ignore', invalid='ignore')
_check = ['average_precision_multi', 'mean_average_precision']

class Metric(object):
    def reset(self):
        pass

    def evaluate(self, oup, target):
        pass

    @property
    def value(self):
        raise NotImplementedError

    @property
    def name(self):
        return class2attr(self, "Metric")


class EvaluateMetric(Metric):
    def __init__(self, func, **kwargs) -> None:
        self.func = func if not isinstance(func, str) else eval(func)
        self.flatten = True
        if self.func.__name__ in _check:
            self.flatten = False
        self.kwargs = kwargs

    def reset(self):
        self.oup, self.target = [], []

    def assemble(self, oup, target):
        if self.flatten:
            self.oup = torch.argmax(oup, dim=-1).cpu().detach().numpy()
            self.target = target.cpu().detach().numpy()
        else:
            self.oup = oup.softmax(dim=-1).cpu().detach().numpy()
            self.target = target.cpu().detach().numpy()

    @property
    def value(self):
        return self.func(self.target, self.oup, **self.kwargs)

    def __call__(self, oup, target):
        self.reset()
        self.assemble(oup, target)
        return self.value

    @property
    def name(self):
        return (
            self.func.func.__name__
            if hasattr(self.func, "func")
            else self.func.__name__
        )


def accuracy(target, oup, **kwargs):
    return metrics.accuracy_score(target, oup, **kwargs)


def f1_score(target, oup, **kwargs):
    return metrics.f1_score(target, oup, **kwargs)


def precision(target, oup, **kwargs):
    return metrics.precision_score(target, oup, **kwargs)


def recall(target, oup, **kwargs):
    return metrics.recall_score(target, oup, **kwargs)


def average_precision_binary(target, oup, **kwargs):
    return metrics.average_precision_score(target, oup, pos_label=1, **kwargs)


def average_precision_multi(target, oup, **kwargs):
    num_classes = oup.shape[-1]
    return np.nan_to_num([
        metrics.average_precision_score(target == i, oup[:, i])
        for i in range(num_classes)
    ])

def mean_average_precision(target, oup, **kwargs):
    num_classes = oup.shape[-1]
    return np.mean(average_precision_multi(target, oup, num_classes=num_classes))


if __name__ == "__main__":
    ev = EvaluateMetric("mean_average_precision")
    l1 = torch.tensor(
        [
            [0.9, 0.2, 0.1],
            [0.3, 0.3, 0.7],
            [0.7, 0.4, 0.3],
            [0.6, 0.5, 0.7],
            [0.6, 0.6, 0.7,],
        ]
    )
    l2 = torch.tensor([0, 2, 0, 2, 0 ])
    # print(sklearn_mean_ap(l1, l2))
    print(ev(l1, l2))
    print(ev.name)

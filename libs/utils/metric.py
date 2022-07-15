import torch


class AccMetric:
    def __call__(self, preds, labels, labels_mask):
        mask = (labels_mask != 0) & (labels != -1)
        correct_nums = float(torch.sum((preds == labels) & mask).detach().cpu().item())
        total_nums = max(float(torch.sum(mask).detach().cpu().item()), 1e-6)
        return correct_nums, total_nums


class AccMulMetric:
    def __call__(self, preds, labels, labels_mask):
        mask = labels_mask != 0
        correct_nums = float(torch.sum((preds == labels).min(1)[0] & mask).detach().cpu().item())
        total_nums = max(float(torch.sum(mask).detach().cpu().item()), 1e-6)
        return correct_nums, total_nums
import torch
from torch import Tensor 

def dice_loss(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):

    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return 1 - dice.mean()

def f1_loss(input: Tensor, target: Tensor, reduce_batch_first: bool = False, thres = 0.25, epsilon: float = 1e-6):

    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first  # 输入应为四维 tensor

    input = (input > thres).float()  # 使用阈值 0.5 进行二值化
    target = (target > thres).float()  # 使用阈值 0.5 进行二值化

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # 计算 True Positives, False Positives 和 False Negatives
    TP = (input * target).sum(dim=sum_dim)  # True Positives
    FP = (input * (1 - target)).sum(dim=sum_dim)  # False Positives
    FN = ((1 - input) * target).sum(dim=sum_dim)  # False Negatives

    # 计算精确率和召回率
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)

    # 计算 F1 分数
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    # 返回 F1 损失
    return 1 - f1_score.mean()
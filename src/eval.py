import torch
def accuracy_score_tensors(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()

def f1_score_tensors(y_true, y_pred):
    intersection = (y_true * y_pred).sum().float()
    precision = intersection / (y_pred.sum().float() + 1e-6)
    recall = intersection / (y_true.sum().float() + 1e-6)
    return 2 * (precision * recall) / (precision + recall + 1e-6)

# predict
def predict_labels(output, threshold=0.25):
    return (output > threshold).type(torch.uint8)
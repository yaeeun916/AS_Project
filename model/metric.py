import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# binary classification
def accuracy(output, target):
    with torch.no_grad():
        # last layer of model is linear layer without sigmoid
        pred = torch.round(torch.sigmoid(output))
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def auroc(output, target):
    output = torch.sigmoid(output)
    return roc_auc_score(target, output)

def precision(output, target):
    pred = torch.round(torch.sigmoid(output))
    return precision_score(target, pred)

def recall(output, target):
    pred = torch.round(torch.sigmoid(output))
    return recall_score(target, pred)

def f1(output, target):
    pred = torch.round(torch.sigmoid(output))
    return f1_score(target, pred)
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

# binary cross entropy with logits
# : combines Sigmoid layer and BCE loss - more numerically stable than using them separately
def bce_with_logits(output, target, pos_weight=None):
    return F.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)

def bce(output, target):
    return F.binary_cross_entropy(output, target)

def cross_entropy(output, target):
    return F.cross_entropy(output, target)
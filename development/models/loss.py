import torch.nn as nn

def define_loss(opt, weights = None):
    if opt.dataset_mode == 'classification':
        loss = nn.CrossEntropyLoss()
    elif opt.dataset_mode == 'segmentation':
        loss = nn.CrossEntropyLoss(ignore_index=-1, weight=weights)
    return loss
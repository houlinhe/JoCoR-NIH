import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def cross_entropy(logits, y_true, num_class, ignore_index = -100):
    # Initialize loss to zero
    loss = 0.0
    loss_log = torch.log( torch.clamp(F.softmax(logits, -1), min=1e-5, max=1.))
    # loss_log = torch.log( torch.clamp(F.sigmoid(logits), min=1e-5, max=1.))
    for i in range(num_class):
        mask = y_true[:,i]!=ignore_index
        if mask.sum(dim=0) > 0:
            loss_mean = ((loss_log[:,i] * y_true[:,i])*mask).sum(dim=0)/mask.sum(dim=0)
            loss += -1 * loss_mean
    return loss


def loss_jocor(y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda=0.1):

    loss_pick_1 = F.cross_entropy(y_1, t, reduce = False) * (1-co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, t, reduce = False) * (1-co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2,reduce=False) + co_lambda * kl_loss_compute(y_2, y_1, reduce=False)).cpu()


    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/float(num_remember)

    ind_update=ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])

    return loss, loss, pure_ratio, pure_ratio

def loss_jocor_no_noise_or_not(y_1, y_2, t, forget_rate, co_lambda=0.1, printornot=0, logger = None, num_class = 3, ind = None, noise_or_not = None):
    tt = F.one_hot(t, num_classes=num_class)
    loss_pick_1 = cross_entropy(y_1, tt, num_class) * (1.0-co_lambda)
    
    loss_pick_2 = cross_entropy(y_2, tt, num_class) * (1.0-co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2,reduce=False) + co_lambda * kl_loss_compute(y_2, y_1, reduce=False))
    loss_pick2 = loss_pick.cpu()

    ind_sorted = np.argsort(loss_pick2.data)
    loss_sorted = loss_pick2[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/float(num_remember)

    ind_update=ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])

    # if printornot == 0:
    # logger.info("")
    # logger.info("Loss Function:")

    # logger.info("loss_pick_1")
    # logger.info(loss_pick_1)
    # logger.info("loss_pick_2")
    # logger.info(loss_pick_2)
    # logger.info("loss_pick")
    # logger.info(loss_pick)
    
    # logger.info("ind_sorted")
    # logger.info(ind_sorted)
    # logger.info(loss_sorted)
    # logger.info("num_remember")
    # logger.info(num_remember)
    # logger.info("ind_update")
    # logger.info(ind_update)

    # logger.info("loss")
    # logger.info(loss)

    # logger.info("pure_ratio, pure_ratio")
    # logger.info(pure_ratio)

    return loss, loss, pure_ratio, pure_ratio

def loss_jocor_no_noise_or_not_nih(y_1, y_2, t, forget_rate, co_lambda=0.1, num_class = 14):
    tt = F.one_hot(t, num_classes=num_class)
    loss_pick_1 = cross_entropy(y_1, tt, num_class) * (1.0-co_lambda)
    
    loss_pick_2 = cross_entropy(y_2, tt, num_class) * (1.0-co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2,reduce=False) + co_lambda * kl_loss_compute(y_2, y_1, reduce=False))
    loss_pick2 = loss_pick.cpu()

    ind_sorted = np.argsort(loss_pick2.data)
    loss_sorted = loss_pick2[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update=ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])

    return loss, loss


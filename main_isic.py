# -*- coding:utf-8 -*-
import os
import torch
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
import argparse, sys
import datetime
from algorithm.jocor import JoCoR
from algorithm.loss import loss_jocor_no_noise_or_not
from utils import ImageDataset
from data_loader.load_isic import load_isic

import logging
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.1)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='mnist')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--co_lambda', type=float, default=0.1)
parser.add_argument('--adjust_lr', type=int, default=1)
parser.add_argument('--model_type', type=str, help='[mlp,cnn]', default='cnn')
parser.add_argument('--save_model', type=str, help='save model?', default="False")
parser.add_argument('--save_result', type=str, help='save result?', default="True")
parser.add_argument('--batch_size', type=int, help='save result?', default=32)



args = parser.parse_args()

logger = logging.getLogger("ydk_logger")
fileHandler = logging.FileHandler('train.log')
streamHandler = logging.StreamHandler()

logger.addHandler(fileHandler)
logger.addHandler(streamHandler)

logger.setLevel(logging.INFO)

# Seed
torch.manual_seed(args.seed)
if args.gpu is not None:
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.manual_seed(args.seed)

else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

print("Device")
print(device)
# Hyper Parameters
batch_size = 32
learning_rate = args.lr / 2
logger.info(args)
## NIH
input_channel = 3
num_classes = 3
init_epoch = 10
args.epoch_decay_start = 100
# args.n_epoch = 200
filter_outlier = False
args.model_type = "cnn"


train_dataset, test_dataset, noise_or_not, actual_noise_rate = load_isic("../ISIC-2017_Training_Data/",
                                        "../ISIC-2017_Test_v2_Data/",
                                        (224, 224), batch_size, args.noise_type, args.noise_rate, logger=logger)
logger.info("noise_or_not, actual_noise_rate")
logger.info(np.array(noise_or_not[0 : 20]))
logger.info(actual_noise_rate)

if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate

def main():
    # Data Loader (Input Pipeline)
    logger.info('loading dataset...')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
											shuffle=True, num_workers=8)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
											shuffle=False, num_workers=8)
    # Define models
    logger.info('building model...')

    args.model_type = "resnet"

    model = JoCoR(args, train_dataset, device, input_channel, num_classes, loss_fn = loss_jocor_no_noise_or_not, linear_layer_size = 12544, logger = logger, noise_or_not = noise_or_not)

    epoch = 0
    train_acc1 = 0
    train_acc2 = 0

    # evaluate models with random weights
    test_acc1, test_acc2 = model.evaluate(test_loader)

    logger.info(
        'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f ' % (
            epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))


    acc_list = []
    # training
    for epoch in range(1, args.n_epoch):
        # train models
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list = model.train_isic(train_loader, epoch)

        # logger.info("pure_ratio_1_list")
        # logger.info(pure_ratio_1_list)
        # logger.info("pure_ratio_2_list")
        # logger.info(pure_ratio_2_list)

        mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)

        # evaluate models
        test_acc1, test_acc2 = model.evaluate(test_loader)

        # save results
        logger.info(
            'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (
                epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1,
                    mean_pure_ratio2))


        if epoch >= 190:
            acc_list.extend([test_acc1, test_acc2])

    avg_acc = sum(acc_list)/len(acc_list)
    logger.info(len(acc_list))
    logger.info("the average acc in last 10 epochs: {}".format(str(avg_acc)))


if __name__ == '__main__':
    main()

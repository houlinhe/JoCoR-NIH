# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.cnn import MLPNet,CNN
from model.resnet import resnet18
import numpy as np
from common.utils import accuracy, accuracy_sig

from algorithm.loss import loss_jocor
import matplotlib.pyplot as plt

class JoCoR:
    def __init__(self, args, train_dataset, device, input_channel, num_classes, loss_fn = None, linear_layer_size = 256, logger = None, noise_or_not = None):

        # Hyper Parameters
        self.batch_size = 32
        learning_rate = args.lr # / 2

        self.train_1 = []
        self.train_2 = []
        self.evalua1 = []
        self.evalua2 = []
        self.roundn = []
        self.round2 = []
        self.round_num = 0
        self.round_num2 = 0

        if args.forget_rate is None:
            if args.noise_type == "asymmetric":
                forget_rate = args.noise_rate / 2
            else:
                forget_rate = args.noise_rate
        else:
            forget_rate = args.forget_rate

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [learning_rate] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch

        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

        print("self.rate_schedule")
        print(self.rate_schedule)

        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        self.co_lambda = args.co_lambda
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset

        if args.model_type == "cnn":
            self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes, linear_layer_size = linear_layer_size)
            self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes, linear_layer_size = linear_layer_size)
        elif args.model_type == "mlp":
            self.model1 = MLPNet()
            self.model2 = MLPNet()
        elif args.model_type == "resnet":
            self.model1 = resnet18(in_channels=input_channel, num_classes=num_classes)
            self.model2 = resnet18(in_channels=input_channel, num_classes=num_classes)

        self.model1.to(device)
        # print(self.model1.parameters)

        self.model2.to(device)
        # print(self.model2.parameters)

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                          lr=learning_rate)

        if loss_fn is not None:
            self.loss_fn = loss_fn
            self.noise_or_not = noise_or_not
        else:
            self.loss_fn = loss_jocor
            self.noise_or_not = train_dataset.noise_or_not

        self.logger = logger

        self.adjust_lr = args.adjust_lr

    # Evaluate the Model
    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        correct1 = 0
        total1 = 0

        mean1 = []
        mean2 = []

        for images, labels, _ in test_loader:
            images = Variable(images).to(self.device)
            logits1 = self.model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()
            mean1.append((pred1.cpu() == labels).sum() / labels.size(0))

        correct2 = 0
        total2 = 0
        for images, labels, _ in test_loader:
            images = Variable(images).to(self.device)
            logits2 = self.model2(images)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)
            total2 += labels.size(0)
            correct2 += (pred2.cpu() == labels).sum()
            mean2.append((pred2.cpu() == labels).sum() / labels.size(0))

        acc1 = 100 * float(correct1) / float(total1)
        acc2 = 100 * float(correct2) / float(total2)

        self.evalua1.append(acc1)
        self.evalua2.append(acc2)
        self.round2.append(self.round_num2)
        self.round_num2 += 1

        plt.plot(self.round2, self.evalua1, color="r")
        plt.savefig("Test_Accuracy1.png")
        plt.clf()
        plt.plot(self.round2, self.evalua2, color="r")
        plt.savefig("Test_Accuracy2.png")
        plt.clf()

        return acc1, acc2

    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        for i, (images, labels, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
            if i > self.num_iter_per_epoch:
                break

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)

            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2(images)
            prec2 = accuracy(logits2, labels, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2

            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch],
                                                                 ind, self.noise_or_not, self.co_lambda)

            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)
            
            print(logits1[0])
            print(logits2[0])
            print(prec1)
            print(prec2)
            print(labels[0])
            # print(prec1[0])
            # print(prec2[0])
            # print(self.model1.c1.weight[0, 0, 0])
            # print(self.model1.c1.weight.grad[0, 0, 0])
            # print(self.model2.c1.weight[0, 0, 0])
            # print(self.model2.c1.weight.grad[0, 0, 0])

            print(
                'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f'
                % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
                loss_1.data.item(), loss_2.data.item()))

            # if (i + 1) % self.print_freq == 0:
            #     print(
            #         'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%'
            #         % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
            #            loss_1.data.item(), loss_2.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list), sum(pure_ratio_2_list) / len(pure_ratio_2_list)))

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)
        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list
    
    def train_isic(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        for i, (images, labels, indexes) in enumerate(train_loader):
            if i > self.num_iter_per_epoch:
                break

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)
            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2(images)
            prec2 = accuracy(logits2, labels, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2

            # self.logger.info("")
            # self.logger.info("Training Info:")
            # self.logger.info(logits1[0])
            # self.logger.info(logits2[0])
            # self.logger.info(labels[0])

            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch], self.co_lambda, i % self.print_freq, self.logger, ind = indexes.cpu().numpy().transpose(), noise_or_not=self.noise_or_not)

            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)

            # if i % self.print_freq == 0:
            self.logger.info("")
            self.logger.info("Training Info 2:")
            self.logger.info(self.model1.layer4[1].conv2.weight[0, 0, 0])
            self.logger.info(self.model1.layer4[1].conv2.weight.grad[0, 0, 0])
            self.logger.info(self.model2.layer4[1].conv2.weight[0, 0, 0])
            self.logger.info(self.model2.layer4[1].conv2.weight.grad[0, 0, 0])

            self.logger.info(
                'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%'
                % (epoch, self.n_epoch, i, len(self.train_dataset) // self.batch_size, prec1, prec2,
                loss_1.data.item(), loss_2.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list), sum(pure_ratio_2_list) / len(pure_ratio_2_list)))
            ###

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)

        self.train_1.append(train_acc1)
        self.train_2.append(train_acc2)
        self.roundn.append(self.round_num)
        self.round_num += 1

        plt.plot(self.roundn, self.train_1, color="r")
        plt.savefig("Train_Accuracy1.png")
        plt.clf()
        plt.plot(self.roundn, self.train_2, color="r")
        plt.savefig("Train_Accuracy2.png")
        plt.clf()

        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list
    
    def train_nih(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        for i, (images, labels, indexes) in enumerate(train_loader):
            if i > self.num_iter_per_epoch:
                break

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)
            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            prec1 = accuracy_sig(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2(images)
            prec2 = accuracy_sig(logits2, labels, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2

            # self.logger.info("")
            # self.logger.info("Training Info:")
            # self.logger.info(logits1[0])
            # self.logger.info(logits2[0])
            # self.logger.info(labels[0])

            loss_1, loss_2 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch], self.co_lambda)

            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            # if i % self.print_freq == 0:
            self.logger.info("")
            self.logger.info("Training Info 2:")
            self.logger.info(self.model1.layer4[1].conv2.weight[0, 0, 0])
            self.logger.info(self.model1.layer4[1].conv2.weight.grad[0, 0, 0])
            self.logger.info(self.model2.layer4[1].conv2.weight[0, 0, 0])
            self.logger.info(self.model2.layer4[1].conv2.weight.grad[0, 0, 0])

            self.logger.info(
                'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f'
                % (epoch, self.n_epoch, i, len(self.train_dataset) // self.batch_size, prec1, prec2,
                loss_1.data.item(), loss_2.data.item()))
            ###

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)

        self.train_1.append(train_acc1)
        self.train_2.append(train_acc2)
        self.roundn.append(self.round_num)
        self.round_num += 1

        plt.plot(self.roundn, self.train_1, color="r")
        plt.savefig("Train_Accuracy1.png")
        plt.clf()
        plt.plot(self.roundn, self.train_2, color="r")
        plt.savefig("Train_Accuracy2.png")
        plt.clf()

        return train_acc1, train_acc2

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1

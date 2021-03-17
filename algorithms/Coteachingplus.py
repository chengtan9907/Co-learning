import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from utils import get_model
from losses import loss_coteaching
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Coteachingplus:
    def __init__(
            self, 
            config: dict = None, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):

        self.lr = config['lr']

        if config['forget_rate'] is None:
            if config['noise_type'] == 'asym':
                forget_rate = config['percent'] / 2
            else:
                forget_rate = config['percent']
        else:
            forget_rate = config['forget_rate']

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [self.lr] * config['epochs']
        self.beta1_plan = [mom1] * config['epochs']

        for i in range(config['epoch_decay_start'], config['epochs']):
            self.alpha_plan[i] = float(config['epochs'] - i) / (config['epochs'] - config['epoch_decay_start']) * self.lr
            self.beta1_plan[i] = mom2

        self.device = device
        self.epochs = config['epochs']

        # define drop rate schedule
        self.rate_schedule = np.ones(config['epochs']) * forget_rate
        self.rate_schedule[:config['num_gradual']] = np.linspace(0, forget_rate ** config['exponent'], config['num_gradual'])

        # model
        self.model1 = get_model(config['model1_type'], input_channel, num_classes, device)
        self.model2 = get_model(config['model2_type'], input_channel, num_classes, device)

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()), lr=self.lr)
        self.adjust_lr = config['adjust_lr']
        self.loss_fn = loss_coteaching

    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        correct1 = 0
        total1 = 0
        for images, labels in test_loader:
            images = Variable(images).to(self.device)
            logits1 = self.model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()

        correct2 = 0
        total2 = 0
        for images, labels in test_loader:
            images = Variable(images).to(self.device)
            logits2 = self.model2(images)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)
            total2 += labels.size(0)
            correct2 += (pred2.cpu() == labels).sum()

        acc1 = 100 * float(correct1) / float(total1)
        acc2 = 100 * float(correct2) / float(total2)
        return acc1, acc2

    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  
        self.model2.train()

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)

        pbar = tqdm(train_loader)
        for (images, labels) in pbar:
            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)

            logits1 = self.model1(images)
            _, pred1 = torch.max(logits1, dim=1)

            logits2 = self.model2(images)
            _, pred2 = torch.max(logits2, dim=1)

            inds = torch.where(pred1 != pred2)
            if len(inds[0]) * (1 - self.rate_schedule[epoch]) < 1:
                loss_1 = F.cross_entropy(logits1, labels)
                loss_2 = F.cross_entropy(logits2, labels)
            else:
                loss_1, loss_2 = self.loss_fn(logits1[inds], logits2[inds], labels[inds], self.rate_schedule[epoch])

            self.optimizer.zero_grad()
            loss_1.backward()
            loss_2.backward()
            self.optimizer.step()

            pbar.set_description(
                    'Epoch [%d/%d], Loss1: %.4f, Loss2: %.4f'
                    % (epoch + 1, self.epochs, loss_1.data.item(), loss_2.data.item()))

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
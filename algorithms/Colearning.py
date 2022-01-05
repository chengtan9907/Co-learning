import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import Model_r18
from tqdm import tqdm
from torch.distributions.beta import Beta

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from losses import loss_structrue, NTXentLoss

# REBUTTAL
from losses import loss_structrue_t

# def D(self, p, z):
#         p = F.normalize(p, p=2, dim=1)
#         z = F.normalize(z, p=2, dim=1)
#         return (p * z).sum(dim=1).mean()

class Colearning:
    def __init__(
            self, 
            config: dict = None, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):

        self.batch_size = config['batch_size']
        self.lr = config['lr']

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

        # scratch
        self.model_scratch = Model_r18(feature_dim=config['feature_dim'], is_linear=True, num_classes=num_classes).to(device)

        self.optimizer1 = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
        self.optimizer2 = torch.optim.Adam(list(self.model_scratch.fc.parameters()), lr=self.lr / 5)
        self.adjust_lr = config['adjust_lr']
        self.ntxent = NTXentLoss(self.device, self.batch_size, temperature=0.5, use_cosine_similarity=True)
        self.param_v = None
        if 'distribution_t' in config.keys():
            self.param_v = config['distribution_t']

    def mixup_data(self, x, y, alpha=5.0):
        lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample() if alpha > 0 else 1
        index = torch.randperm(x.size()[0]).cuda() 
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam=0.5):
        return (lam * F.cross_entropy(pred, y_a, reduce=False) + (1 - lam) * F.cross_entropy(pred, y_b, reduce=False)).mean()

    def evaluate(self, test_loader):
        print('Evaluating ...')

        self.model_scratch.eval()  # Change model to 'eval' mode

        correct2 = 0
        total2 = 0
        for images, labels in test_loader:
            images = Variable(images).to(self.device)
            _, _, logits2 = self.model_scratch(images)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)
            total2 += labels.size(0)
            correct2 += (pred2.cpu() == labels).sum()

        acc2 = 100 * float(correct2) / float(total2)
        return acc2

    def train(self, train_loader, epoch):
        print('Training ...')

        self.model_scratch.train()

        if self.adjust_lr:
            self.adjust_learning_rate(self.optimizer1, epoch)
            self.adjust_learning_rate(self.optimizer2, epoch)

        pbar = tqdm(train_loader)
        for item in pbar:
            raw, pos_1, pos_2, labels = item[0:4]
            pos_1, pos_2 = Variable(pos_1).to(self.device, non_blocking=True), Variable(pos_2).to(self.device, non_blocking=True)
            labels = Variable(labels).to(self.device)
            raw = Variable(raw).to(self.device, non_blocking=True)

            feat, outs, logits = self.model_scratch(raw)
            if self.param_v is None:
                loss_feat = loss_structrue(outs.detach(), logits)
            else:
                loss_feat = loss_structrue_t(outs.detach(), logits, self.param_v)
            self.optimizer2.zero_grad()
            loss_feat.backward()
            self.optimizer2.step()

            # Self-learning
            out_1 = self.model_scratch(pos_1, ignore_feat=True, forward_fc=False)
            out_2 = self.model_scratch(pos_2, ignore_feat=True, forward_fc=False)
            loss_con = self.ntxent(out_1, out_2)

            feat, outs, logits = self.model_scratch(raw)

            # Supervised-learning
            inputs, targets_a, targets_b, lam = self.mixup_data(raw, labels, alpha=5.0)
            _, logits = self.model_scratch(inputs, ignore_feat=True)
            loss_sup = self.mixup_criterion(logits, targets_a, targets_b, lam)

            # Loss
            loss = loss_sup + loss_con 

            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()

            pbar.set_description(
                    'Epoch [%d/%d], loss_con: %.4f, loss_sup: %.4f'
                    % (epoch + 1, self.epochs, loss_con.data.item(), loss_sup.data.item()))


    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1

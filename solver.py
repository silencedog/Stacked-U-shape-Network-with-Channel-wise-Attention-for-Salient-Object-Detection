import os
import cv2
import math
import torch
from loss import Loss
import scipy.io as io
from network_normal import build_model
from torch.optim import Adam
from torch.backends import cudnn
from torchvision import transforms
from torch.nn import utils, functional as F
import numpy as np
np.set_printoptions(threshold=np.nan)



class Solver(object):
    def __init__(self, train_loader, val_loader, test_dataset, config):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_dataset = test_dataset
        self.config       = config
        self.beta         = math.sqrt(0.3)
        self.lr_decay_epoch = [80, 90]
        self.device = torch.device('cpu')
        self.mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)#view()函数作用是将一个多行的Tensor,拼接成某种行
        self.std    = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.visual_save_fold = config.pre_map
        if self.config.cuda:
            cudnn.benchmark = True
            self.device     = torch.device('cuda:0')
        self.build_model()
        if self.config.pre_trained: self.net.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'train':
            self.log_output  = open("%s/logs/log.txt" % config.save_fold, 'w')
        else:
            self.net.load_state_dict(torch.load(self.config.model))
            self.net.eval()
            self.test_output = open("%s/test.txt" % config.test_fold, 'w')
            self.transform   = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # build the network
    def build_model(self):
        self.net = build_model().to(self.device)
        if self.config.mode == 'train': self.loss = Loss().to(self.device)
        self.net.train()
        self.optimizer = Adam(self.net.parameters(), lr=self.config.lr)

    # evaluate MAE (for test or validation phase)
    def eval_mae(self, y_pred, y):

        return torch.abs(y_pred - y).mean()

    # validation: using resize image, and only evaluate the MAE metric
    def validation(self):
        avg_mae = 0.0
        self.net.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(self.val_loader):
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                prob_pred      = self.net(images)
                avg_mae       += self.eval_mae(prob_pred[0], labels).item()
        self.net.train()
        return avg_mae / len(self.val_loader)

    # test phase: using origin image size, evaluate MAE and max F_beta metrics
    def test(self, num, use_crf=False):
        avg_mae, img_num = 0.0, len(self.test_dataset)

        with torch.no_grad():
            for i, (img, labels, name) in enumerate(self.test_dataset):
                images = self.transform(img).unsqueeze(0)
                labels = labels.unsqueeze(0)
                shape  = labels.size()[2:]
                images = images.to(self.device)

                prob_pred, block0, block1, block2, block3, block4 = self.net(images)  #因为输出多个 测试的时候需要改一下

                block0 = np.array(block0[0, :, :, :].cpu())
                io.savemat('block0.mat', {'block0': block0})
                block1 = np.array(block1[0, :, :, :].cpu())
                io.savemat('block1.mat', {'block1': block1})
                block2 = np.array(block2[0, :, :, :].cpu())
                io.savemat('block2.mat', {'block2': block2})
                block3 = np.array(block3[0, :, :, :].cpu())
                io.savemat('block3.mat', {'block3': block3})
                block4 = np.array(block4[0, :, :, :].cpu())
                io.savemat('block4.mat', {'block4': block4})


                prob_pred = F.interpolate(prob_pred[0], size=shape, mode='bilinear', align_corners=True).cpu().data
   
                if not os.path.exists('{}/'.format(self.visual_save_fold)):
                    os.mkdir('{}/'.format(self.visual_save_fold))
             
                img_save = prob_pred.numpy()
                img_save = img_save.reshape(-1, img_save.shape[2], img_save.shape[3]).transpose(1, 2, 0) * 255
                cv2.imwrite('{}/{}.png'.format(self.visual_save_fold, name), img_save.astype(np.uint8))

                mae           = self.eval_mae(prob_pred, labels)
                print("[%d] mae: %.4f" % (i, mae))
                print("[%d] mae: %.4f" % (i, mae), file=self.test_output)
                avg_mae += mae
        avg_mae = avg_mae / img_num
        print('average mae: %.4f' % (avg_mae))
        print('average mae: %.4f' % (avg_mae), file=self.test_output)
        

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) / self.config.batch_size
        best_mae = 1.0 if self.config.val else None

        for epoch in range(self.config.epoch):   
            loss_epoch = 0
            for i, data_batch in enumerate(self.train_loader):
                if (i + 1) > iter_num: break
                self.net.zero_grad()
                x, y     = data_batch
                x, y     = x.to(self.device), y.to(self.device)
                y_pred   = self.net(x)
                loss     = self.loss(y_pred, y)
                loss.backward()
                utils.clip_grad_norm_(self.net.parameters(), self.config.clip_gradient)
                self.optimizer.step()
                loss_epoch += loss.item()
                print('epoch: [%d/%d], iter: [%d/%d], loss: [%.4f], lr: [%s]' % (
                    epoch, self.config.epoch, i, iter_num, loss.item(), self.config.lr))

                
            if (epoch + 1) % self.config.epoch_show == 0:
                print('epoch: [%d/%d], epoch_loss: [%.4f], lr: [%s]' % (epoch, self.config.epoch, loss_epoch / iter_num, self.config.lr), file=self.log_output)

            if self.config.val and (epoch + 1) % self.config.epoch_val == 0:
                mae = self.validation()
                print('--- Best MAE: %.5f, Curr MAE: %.5f ---' % (best_mae, mae))
                print('--- Best MAE: %.5f, Curr MAE: %.5f ---' % (best_mae, mae), file=self.log_output)
                if best_mae > mae:
                    best_mae = mae
                    torch.save(self.net.state_dict(), '%s/models/best.pth' % self.config.save_fold)
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_fold, epoch + 1))
            if epoch in self.lr_decay_epoch:
                self.config.lr  = self.config.lr * 0.1
                self.optimizer = Adam(self.net.parameters(), lr=self.config.lr)
        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_fold)     

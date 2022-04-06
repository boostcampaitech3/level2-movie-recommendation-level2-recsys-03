'''
/**
 * Original Code
 * https://github.com/pyy0715/Neural-Collaborative-Filtering
 * modified by Ye-ji Kim
 */
 '''

from tqdm import tqdm
import os

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from util import metrics


class Trainer():
    def __init__(self, model, optimizer, loss_function, config) :
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.config = config
        self.writer = SummaryWriter()

        super().__init__()

        self.device = next(model.parameters()).device

    def _train(self, train_loader) :
        # Turn train mode on.
        self.model.train()
        total_loss = 0

        for user, item, label in tqdm(train_loader):
            user = user.to(self.device)
            item = item.to(self.device)
            label = label.to(self.device)

            prediction = self.model(user, item)
            loss = self.loss_function(prediction, label)

            # Initialize the gradients of the model.
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            total_loss += float(loss)

        return total_loss/len(train_loader)

    
    def _validate(self, valid_loader):
        # Turn evaluation mode on.
        self.model.eval()  
        total_loss = 0    

        with torch.no_grad():
            for user, item, label in valid_loader:
                user = user.to(self.device)
                item = item.to(self.device)
                label = label.to(self.device)

                prediction = self.model(user, item)
                loss = self.loss_function(prediction, label)

                total_loss += float(loss)
            
            HR, NDCG = metrics(self.model, valid_loader, self.config.top_k, self.device)

        return total_loss/len(valid_loader), HR, NDCG
    
    
    def train(self, train_loader, valid_loader):
        best_hr = 0

        for epoch in tqdm(range(self.config.n_epochs)):
            train_loss = self._train(train_loader)
            valid_loss, HR, NDCG = self._validate(valid_loader)

            self.writer.add_scalar('loss/Train_loss', train_loss, epoch)
            self.writer.add_scalar('loss/Valid_loss', valid_loss, epoch)
            self.writer.add_scalar('Perfomance/HR@10', HR, epoch)
            self.writer.add_scalar('Perfomance/NDCG@10', NDCG, epoch)

            print("Epoch(%d/%d): train_loss=%.4f  valid_loss=%.4f  HR=%.4f  NDCG=%.4f" % (
                epoch + 1,
                self.config.n_epochs,
                train_loss,
                valid_loss,
                np.mean(HR),
                np.mean(NDCG),
            ))
            
            # Save the best model.
            if HR > best_hr:
                best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
                if self.config.out:
                    if not os.path.exists(self.config.MODEL_PATH):
                        os.mkdir(self.config.MODEL_PATH)

                    output_path = os.path.join(self.config.MODEL_PATH, self.config.MODEL)
                    torch.save(self.model, output_path)

        self.writer.close()
        print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
                                            best_epoch, best_hr, best_ndcg))
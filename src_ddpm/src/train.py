from utils3 import *
from optimizer import loss_fun, optimizer
from dataLoader import scoreloader
from model2 import *
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

import os, time, datetime
from tqdm import tqdm
from test import scoretester
import torchvision.utils
import shutil

class scoretrainer():
    def __init__(self):
        super(scoretrainer, self).__init__()
        parser = config_parser()
        self.args = parser.parse_args()
        print('config args:', self.args)

        if not self.args.expname:
            self.args.expname  = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.expdir = os.path.join(self.args.logsdir, self.args.expname)

        # loader = scoreloader(projdir=self.args.datadir,
        #                      psize = self.args.patchsize, sigma = self.args.sigma)
        
        
        # num_valid = int(len(loader) * self.args.validfrac)
        # train, val = data.random_split(loader, [len(loader) - num_valid, num_valid],
        #                                generator=torch.Generator().manual_seed(42))
        # instead of splitting the data, make 2 loaders.
        ####################NOTE#######################
        # add "traindatadir" and "valdatadir" to parser
        ####################NOTE#######################
        train_loader = scoreloader(projdir=self.args.traindatadir,
                                    psize = self.args.patchsize, 
                                    sigma = self.args.sigma)
        self.trainloader = data.DataLoader(train_loader,
                                          batch_size=self.args.batchsize,
                                          shuffle=True,
                                          num_workers=8,
                                          pin_memory=True)
        self.psize, self.lsize = train_loader.getsize()
        # valid loader
        valid_loader = scoreloader(projdir=self.args.valdatadir,
                                    psize = self.args.patchsize, 
                                    sigma = self.args.sigma)
        
        self.validloader = data.DataLoader(valid_loader,
                                           batch_size=self.args.batchsize,
                                           shuffle=True,
                                           num_workers=8,
                                           pin_memory=True)
        
        #self.model = Fern(D = self.args.netdepth, W = self.args.netwidth, input_ch = self.psize**2,
        #output_ch = self.psize**2)
        self.model = Unet(dim = self.lsize)

        ##################################################
        # init the gpu usages
        ##################################################
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_ids
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        ##################################################
        # init the training settings
        ##################################################
        self.model.to(self.device)
        self.optimizer = optimizer(self.model, self.args.lr, self.args.optim)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, verbose=True, factor=0.2)
        self.writer = SummaryWriter(self.expdir)
        self.criterion = loss_fun(self.args.loss_fun)
        
        ##################################################
        # record code data
        ##################################################
        copytree_code( os.path.join(self.args.srcdir), self.expdir + '/')
        shutil.copyfile("../config/params.txt", os.path.join(self.expdir, "params.txt"))

    def train(self):
        test_init_done, train_init_done = False, False
        # write config to the tensorboard 
        self.writer.add_text('init/config', str(self.args))
        
        valid_loss_history = []
        Initial_loss = self.valid()
        print('Initial valid loss: ', Initial_loss)
        for epoch in range(self.args.nepochs):
            print('')
            self.model.train()
            start_t = time.time()
            train_loss = 0
            train_snr = 0
            niter = 0
            for batch in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                for key in batch:
                    # print('key: {}, maximum value: {}'.format(key, torch.max(batch[key]).item()))
                    batch[key] = batch[key].to(self.device)
                pred = self.model.forward(batch['input'], batch['time'])
                loss = self.criterion(pred, batch['output'])
                train_loss += loss.item()

                snr_val = snr(batch['output'].cpu().detach().numpy(), pred.cpu().detach().numpy())
                train_snr += snr_val

                loss.backward()
                self.optimizer.step()
                niter += 1

            end_t = time.time()
            train_loss = train_loss / niter
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('SNR/train', train_snr, epoch)

            valid_loss, valid_snr = self.valid()
            valid_loss_history.append(valid_loss)
            self.writer.add_scalar('Loss/valid', valid_loss, epoch)
            self.writer.add_scalar('SNR/valid', valid_snr , epoch) 

            print('(Epoch {} / {}) train loss: {:.4f} valid loss: {:.4f}, train snr: {:.2f}, valid snr: {:.2f},  time per epoch: {:.1f}s current lr: {}'.format(
                epoch + 1,
                self.args.nepochs,
                train_loss,
                valid_loss,
                train_snr,
                valid_snr,
                end_t - start_t,
                self.optimizer.param_groups[0]['lr']))
            if (epoch + 1) % 5 == 0:
                print('Save the current model to checkpoint!')
                save_checkpoint(self.model.state_dict(), is_best=False,
                                checkpoint_dir=f'{self.expdir}/models')
            if epoch == np.argmin(valid_loss_history):
                print('The current model is the best model! Save it!')
                save_checkpoint(self.model.state_dict(), is_best=True,
                                checkpoint_dir=f'{self.expdir}/models')
            self.lr_scheduler.step(valid_loss)

            # do online test for visiualization
            # if self.args.do_online_test and epoch % self.args.online_test_epoch_gap == 0:
            #     test_tester = scoretester(self.model, 
            #                               is_best=False, 
            #                               verbose=False, 
            #                               args=self.args)
            #     train_tester = scoretester(self.model, 
            #                               is_best=False, 
            #                               verbose=False, 
            #                               args=self.args)
                # print(len(self.tester.testloader))
                #self.check_rest(test_tester, epoch, test_init_done, dataset_key='test')
                #self.check_rest(train_tester, epoch, train_init_done, dataset_key='train')
                # test_init_done = True
                # train_init_done = True
        self.writer.close()
    
    def check_rest(self, tester, epoch, init_done):
        pass

    def valid(self):
        self.model.eval()
        with torch.no_grad():
            valid_loss = 0
            valid_snr = 0
            niter = 0
            for batch in self.validloader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                #print(batch['input'].size())
                pred = self.model.forward(batch['input'], batch['time'])
                loss = self.criterion(pred, batch['output'])
                valid_loss += loss.item()

                snr_val = snr(batch['output'].cpu().detach().numpy(), pred.cpu().detach().numpy())
                valid_snr += snr_val

            niter += 1
        return valid_loss / niter, valid_snr/niter
    
if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"]="3"
    trainer = scoretrainer()
    trainer.train()
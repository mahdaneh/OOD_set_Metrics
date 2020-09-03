from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
import os
from configparser import *
import tensorboardX
import torchvision

class Local_OP ():
    def __init__(self, config_file):

        dir = config_file.split('/')[:-1]
        self.conf_fname = config_file.split('/')[-1]

        if len(dir) ==0 : self.dir= os.getcwd()
        else:
            self.dir = os.path.join(*dir)

        print (self.conf_fname, self.dir)

        # parsing the configuration file
        config = ConfigParser()
        config.read(config_file)
        print (config)
        self.dataset_name = config.get('EXPERIMENT_SETTINGS', 'dataset_name')

        self.tr_smpl = config.getint('EXPERIMENT_SETTINGS', 'tr_smpl')
        self.test_smpl = config.getint('EXPERIMENT_SETTINGS', 'test_smpl')

        self.epochs = config.getint('EXPERIMENT_SETTINGS', 'epochs')
        self.batch_size = config.getint('EXPERIMENT_SETTINGS', 'batch_size')
        self.lr = config.getfloat('EXPERIMENT_SETTINGS', 'lr')
        self.momentum = config.getfloat('EXPERIMENT_SETTINGS', 'momentum')

        self.weight_decay = config.getfloat('EXPERIMENT_SETTINGS', 'weight_decay')
        self.num_chanl = config.getint('EXPERIMENT_SETTINGS', 'num_chanl')
        self.image_size = config.getint('EXPERIMENT_SETTINGS', 'image_size')
        self.num_class = config.getint('EXPERIMENT_SETTINGS', 'num_class')
        self.dustbin = config.getboolean('EXPERIMENT_SETTINGS', 'dustbin')
        self.out_dist_flag = config.getboolean('EXPERIMENT_SETTINGS', 'out_dist_flag')
        self.out_data_name = config.get('EXPERIMENT_SETTINGS', 'out_name')
        if not self.out_dist_flag: self.out_data_name = None
        print (self.dustbin, self.image_size, self.num_chanl)
        self.pre_net = config.get('EXPERIMENT_SETTINGS', 'pre_net')

        try:self.checkpoint = config.get('EXPERIMENT_SETTINGS', 'checkpoint')
        except: self.checkpoint=None
        self.writer = tensorboardX.SummaryWriter('runs_'+self.dataset_name)





    def train(self, model, device, train_loader, optimizer, epoch):

        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            # grid_img = torchvision.utils.make_grid(data, nrow=10, padding=30)
            # self.writer.add_image('train_images', grid_img)
            data, target = data.to(device), target.to(device)

            loss_b = self.optimizing(optimizer,model,data,target)
            total_loss+= loss_b

        print('Train Epoch: {} \tLoss_b: {:.6f} \t'.format(epoch, total_loss.item()/batch_idx))

    def train_uniform(self, model, device, train_loader, optimizer, epoch):

        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            loss_b = self.optimizing_unifrom(optimizer, model, data, target, device)
            total_loss += loss_b

        print('Train Epoch: {} \tLoss_b: {:.6f} \t'.format(epoch, total_loss.item() / batch_idx))



    def optimizing(self,optimizer, model, data, target):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy( output, target)
        loss.backward()
        optimizer.step()
        return loss


    # optimization function to train an end-to-end calibrated vanilla CNN [reference: Deep Anomaly Detection with Outlier Exposure; Hendrycks, Dan and Mazeika, Mantas and Dietterich, Thomas]
    def optimizing_unifrom (self,optimizer, model, data, target, device):
        log_softmax = torch.nn.LogSoftmax(dim =1)
        data=data.to(device)
        output = model(data)
        optimizer.zero_grad()


        # non-dustbin class
        indices = (target != self.num_class).nonzero().flatten()
        t1 = target[indices].to(device)

        loss = F.cross_entropy(output[indices,:],t1)

        # dustbin class
        indices = (target == self.num_class).nonzero().flatten()
        if len(indices)>0:
            t2 = output[indices,:].to(device)
            l2=  -(1/self.num_class)* (torch.sum(log_softmax(t2), dim = 1)).mean()
            loss +=l2

        loss.backward()
        optimizer.step()
        return loss


    def test(self, model, device, test_loader, set,save_fea=False,target_dustbin=False):
        # import pdb;pdb.set_trace()
        ####
        model.eval()
        ####
        conf = torch.nn.Softmax(dim=1)
        test_loss = 0
        correct = 0
        confidence=0

        batch = 0
        all_features = [[],[]]
        all_targets = []
        all_y_estim = []
        with torch.no_grad():
            for i_batch , (data, target) in enumerate(test_loader):
                if i_batch==0:
                    grid_img = torchvision.utils.make_grid(data[:3], nrow=3, padding=10)
                    self.writer.add_image('Images_test '+str(set) , grid_img)
                batch+=1
                if self.dustbin and target_dustbin:
                   target = self.num_class*(torch.ones(len(data),dtype=torch.long))
                elif self.dustbin==False and target_dustbin:
                   target = 0*(torch.ones(len(data),dtype=torch.long))


                data, target = data.to(device), target.to(device)
                output = model(data)


                test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss

                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                # if (i_batch + 1) % 100 == 0:
                #     print(target,pred.view_as(target))
                confidence += torch.mean(torch.max(conf(output),dim=1)[0]).item()

                correct += pred.eq(target.view_as(pred)).sum().item()
                if save_fea:
                    # assert  feature_ext is not None
                    # feature = feature_ext['feature_extractor'].view(len(data), -1)
                    _, feature_list = model.feature_list(data)
                    # before FC layers
                    feature = feature_list[-1].view(len(data),-1)

                    all_features[0].append(feature.data.cpu())

                    # penultimate layer
                    if len(feature_list)>1:
                        feature = feature_list[-2].view(len(data),-1)
                        all_features[1].append(feature.data.cpu())
                        all_targets.append(target.data.cpu())
                        all_y_estim.append(pred.data.cpu())


        if save_fea :

            all_features = [np.concatenate(all_features[i], axis=0)  for i in range(len(all_features))]
            all_y_estim = np.concatenate(all_y_estim, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            torch.save({'features' : all_features,'targets' : all_targets,'estimation':all_y_estim }, self.dir+'/feature_space/'+self.conf_fname+str(set))

        test_loss /= len(test_loader.dataset)

        print('\n{:s} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%) Avg confidence {:4f}\n'.format(set,
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), confidence/batch))
        print(len(test_loader.dataset))
        return 100. * correct / len(test_loader.dataset), (confidence/batch)*100


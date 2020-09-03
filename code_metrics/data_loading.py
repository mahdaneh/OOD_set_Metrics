"""TODO: docstring
"""
import torch
import util
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import  Image
import numpy as np
import shutil
import tarfile

from skimage.filters import gaussian as gblur
from astropy.io import ascii




def bechmark_loading ( num_tr_smpl,num_test_smpl, dt_name , our_transform, slct_trgt=None,num_chanl=3, dir= None, test_transform = None, image_size=32, num_class=10):
    if dir is None: exit('locate data directory first!!')

    source_dataset, source_dataset_test, source_dataset_validation = [], [] , []



    print ('==========  loading the current task '+dt_name)
    if dt_name =='mnist':
        source_dataset = (util.Local_Dataset_digit(data_name='mnist', set='train', data_path='../data/mnist', transform=our_transform,
                             num_samples=num_tr_smpl,slct_trgt=slct_trgt))
        source_dataset_test = util.Local_Dataset_digit(data_name='mnist', set='test', data_path='../data/mnist', transform=test_transform,
                                     num_samples=num_test_smpl, slct_trgt = slct_trgt)
        source_dataset_validation= util.Local_Dataset_digit(data_name='mnist', set='validation', data_path='../data/mnist',
                                     transform=our_transform,
                                     num_samples=1000)
    if dt_name == 'm-mnist':

        source_dataset= util.Local_Dataset_digit(data_name='m-mnist', set='train', data_path='../data/mnist',
                                                       transform=our_transform,
                                                       num_samples=num_tr_smpl, slct_trgt=slct_trgt,num_chann=3)

        source_dataset_test = util.Local_Dataset_digit(data_name='m-mnist', set='test', data_path='../data/mnist', transform=test_transform,
                                     num_samples=num_test_smpl,slct_trgt=slct_trgt,num_chann=3)

        source_dataset_validation= util.Local_Dataset_digit(data_name='m-mnist', set='validation', data_path='../data/mnist',
                                     transform=test_transform,
                                     num_samples=1000, slct_trgt=slct_trgt, num_chann=3)
    if dt_name =='usps':
        source_dataset=(util.Local_Dataset_digit(data_name='usps', set='train', data_path='../data/USPSdata',
                                                       transform=our_transform,
                                                       num_samples=num_tr_smpl, num_chann=num_chanl))
        source_dataset_test=(
            util.Local_Dataset_digit(data_name='usps', set='test', data_path='../data/USPSdata', transform=our_transform,
                                     num_samples=num_test_smpl, num_chann=num_chanl))
        source_dataset_validation = (
            util.Local_Dataset_digit(data_name='usps', set='validation', data_path='../data/USPSdata',
                                     transform=our_transform,
                                     num_samples=1000, num_chann=num_chanl))

        print (source_dataset.__len__())
    if dt_name =='svhn':


        source_dataset=(util.Local_SVHN(root=dir+'/SVHN', split='train', transform=our_transform, download=True,
                        num_smpl=num_tr_smpl, num_chanl= num_chanl,slct_trgt=slct_trgt))
        source_dataset_test=(
            util.Local_SVHN(root=dir+'/SVHN', split='test', transform=test_transform, download=True,
                            num_smpl=num_test_smpl, num_chanl=num_chanl,slct_trgt=slct_trgt))

        if num_chanl >1 : print ('RGB svhn loaded')
        else: print('Gray svhn loaded')

    if dt_name =='cifar10':

        source_dataset = util.Local_cifar10(root=dir+'/cifar10', split='train', transform=our_transform, download=True, num_smpl=num_tr_smpl,
                                            slct_trgt=slct_trgt, num_chanl= num_chanl)

        source_dataset_test = util.Local_cifar10(root=dir+'/cifar10', split='test', transform=test_transform, num_smpl=num_test_smpl, slct_trgt=slct_trgt, num_chanl= num_chanl)

    if dt_name =='cifar100':
        source_dataset = util.Local_cifar100(root=dir + '/cifar100', split='train', transform=our_transform,
                                            download=True, num_smpl=num_tr_smpl,
                                            slct_trgt=slct_trgt, num_chanl=num_chanl)

        source_dataset_test = util.Local_cifar100(root=dir + '/cifar100', split='test', transform=test_transform, download=True,
                                                 num_smpl=num_test_smpl, slct_trgt=slct_trgt, num_chanl=num_chanl)

        print(source_dataset.__repr__())



    if dt_name == 'sbu':
        source_dataset = datasets.sbu(dir+'/SBU', transform=our_transform, target_transform=None, download=True)
    if dt_name =='c100_nover':

        source_dataset = util.Local_CIFAR100_noveralp(dir+'/cifar100', split='train',transform=our_transform, target_transform=None, download=False, num_smpl = num_tr_smpl)
        source_dataset_test = util.Local_CIFAR100_noveralp(root=dir + '/cifar100', split='test', transform=test_transform,
                                                  download=False,
                                                  num_smpl=num_test_smpl)

    if dt_name == 'imagnet':

        dataroot = dir + '/Imagenet_resize'
        source_dataset = datasets.ImageFolder(dataroot, transform=our_transform)
        source_dataset.samples = [(d, torch.tensor(num_class)) for d, s in source_dataset.samples]

        source_dataset, source_dataset_test = util.splitting_dataset(source_dataset, te_sample_ratio=0)



    if dt_name == 'lsun':
        dataroot = dir + '/LSUN_resize'
        source_dataset = datasets.ImageFolder(dataroot, transform=our_transform)
        source_dataset.samples = [(d, torch.tensor(num_class)) for d, s in source_dataset.samples]

        source_dataset, source_dataset_test = util.splitting_dataset(source_dataset, te_sample_ratio=0)



    if dt_name == 'isun':
        dataroot = dir + '/iSUN'
        # source_dataset = util.ImageFolder_local(dataroot, transform=our_transform)
        source_dataset = datasets.ImageFolder(dataroot, transform=our_transform)

        source_dataset.samples = [(d, torch.tensor(num_class)) for d, s in source_dataset.samples]
        source_dataset, source_dataset_test = util.splitting_dataset(source_dataset, te_sample_ratio=0)

    if dt_name == 'isun_cropped':
        dataroot = dir + '/iSUN'
        source_dataset = util.ImageFolder_local(dataroot, transform=our_transform)

        source_dataset.samples = [(d, num_class) for d, s in source_dataset.samples]
        source_dataset, source_dataset_test = util.splitting_dataset(source_dataset, te_sample_ratio=0)



    if dt_name=='Gaussian':
        dummy_targets = (torch.ones(num_tr_smpl+num_test_smpl)*num_class).type(torch.long)
        ood_data = torch.from_numpy(np.clip(np.random.normal(size=(num_tr_smpl+num_test_smpl, num_chanl, image_size, image_size),loc=0.5, scale=0.5), 0, 1))

        ood_data = ood_data.type(torch.FloatTensor)
        source_dataset = torch.utils.data.TensorDataset(ood_data[:num_tr_smpl], dummy_targets[:num_tr_smpl])
        source_dataset_test = torch.utils.data.TensorDataset(ood_data[-num_test_smpl:], dummy_targets[-num_test_smpl:])



    if dt_name =='Bernouli':
        dummy_targets = torch.ones(num_tr_smpl+num_test_smpl)*num_class
        ood_data = torch.from_numpy(np.random.binomial(
            n=1, p=0.5, size=(num_tr_smpl+num_test_smpl, num_chanl, image_size, image_size)).astype(np.float32))
        source_dataset = torch.utils.data.TensorDataset(ood_data[:num_tr_smpl], dummy_targets[:num_tr_smpl])
        source_dataset_test = torch.utils.data.TensorDataset(ood_data[-num_test_smpl:], dummy_targets[-num_test_smpl:])

    if dt_name =='Blobs':

        ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(num_tr_smpl+num_test_smpl, num_chanl, image_size, image_size)))
        for i in range(num_tr_smpl):
            ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
            ood_data[i][ood_data[i] < 0.75] = 0.0

        dummy_targets = (torch.ones(num_tr_smpl + num_test_smpl) * num_class).type(torch.long)
        ood_data = torch.from_numpy(ood_data)

        source_dataset = torch.utils.data.TensorDataset(ood_data[:num_tr_smpl], dummy_targets[:num_tr_smpl])
        source_dataset_test = torch.utils.data.TensorDataset(ood_data[-num_test_smpl:], dummy_targets[-num_test_smpl:])

    print('training set size : ',len(source_dataset), 'test set size : ',len(source_dataset_test))

    return source_dataset, source_dataset_test, source_dataset_validation






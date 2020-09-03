from __future__ import print_function
import argparse
import numpy as np
import torch
import data_loading as db

import model_building as builder

import training_op as op
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import compute_covg as covg_metric


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config-file', type=str, default=None)

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    local_op = op.Local_OP(args.config_file)
    train_transform = None
    num_class = local_op.num_class

    if local_op.dataset_name == 'cifar10':
        model = builder.vgg11_bn(num_class).to(device)

    elif local_op.dataset_name == 'svhn':
        model = builder.vgg16_bn(num_class).to(device)

    # model sent to device
    model = model.to(device)

    #  training and test data pre-processing (e.g. mean normalization or resizing input image)
    if train_transform is None:
        train_transform = transforms.Compose(
            [transforms.Resize((local_op.image_size, local_op.image_size)), transforms.ToTensor()])
        test_transform = train_transform

    # db.bechmark_loading( #training samples, #test samples, dataset name, training data transform, #image channel, data directory, test data transfer)
    tr_dataset, _, _ = db.bechmark_loading(local_op.tr_smpl, local_op.test_smpl,
                                                                       local_op.dataset_name, train_transform,
                                                                       num_chanl=local_op.num_chanl, dir='../data',
                                                                       test_transform=test_transform)

    train_loader = DataLoader(tr_dataset, batch_size=local_op.batch_size, shuffle=True, num_workers=0)

    # LOAD a pre-trained network from local_op.dir
    chpnt = torch.load(os.path.join(local_op.dir, local_op.pre_net))
    if 'model' in chpnt.keys():

        model.load_state_dict(chpnt['model'])
    else:
        model.load_state_dict(chpnt)

        # save training samples in the feature space (save_fea=True) into a file (a torch file)
        train_accuracy, _ = local_op.test(model, device, train_loader, 'train', save_fea=True)

        protection_level_measurement(local_op, test_transform, model, device)



def protection_level_measurement(local_op, our_transform, model, device):
    """
    I) load an OOD set ,
    II) save its data in the feature space,
    III) compute the metrics (SE, CR, CD) to reveal the protection level,
    IV) Write the results in a table
    """

    result_dic = []

    Infeature_file = local_op.dir + '/feature_space/' + local_op.conf_fname + 'train'

    if local_op.dataset_name == 'svhn': list_out = ['Gaussian','Blobs','cifar10','lsun', 'isun', 'cifar100', 'imagnet']
    elif local_op.dataset_name =='cifar10':list_out = ['isun','c100_nover','lsun',  'imagnet','svhn' ,'Gaussian','Blobs',]

    for ii, name in enumerate(list_out):

        #  I) loading an OOD set and splitting the data into test and training sets
        train_out_dist, test_out_dist, _ = db.bechmark_loading( num_tr_smpl = 10000, num_test_smpl = 10000,
                                           dt_name = name, our_transform = our_transform, num_chanl=3, dir ='../data', test_transform=our_transform, image_size=local_op.image_size)


        train_out_loader = DataLoader(train_out_dist, batch_size=local_op.batch_size, shuffle=False, num_workers=0)

        # II) to save the OOD training data in the feature space (save_fea = True)
        local_op.test(model, device, train_out_loader, name, save_fea=True,target_dustbin=True)
        Outfeature_file = local_op.dir + '/feature_space/' + local_op.conf_fname + str(name)

        # III) computing metrics to reveal the protection level of each OOD set
        result_dic, in_fea, out_fea = covg_metric.metrics(Infeature_file, Outfeature_file, KNN=list(np.arange(2,40,4))+list(np.arange(40,100,5)), dictionary = result_dic, name = os.path.join(local_op.dir,name), path=local_op.dir)

    ascii.write(result_dic, local_op.dir + '/' + 'Final_Table' + local_op.conf_fname + '.csv',
            format='csv', fast_writer=False)

    return result_dic


def lr_print(optimizer):
    for param_group in optimizer.param_groups:
        print('learning rate ' + str(param_group['lr']))


if __name__ == '__main__':
    main()

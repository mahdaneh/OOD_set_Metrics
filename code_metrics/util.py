"""TODO: docstring
"""

import os
import sys
import shutil
import tarfile
import urllib.request
import zipfile
from PIL import  Image
from skimage.transform import resize as skresize
from skimage.io import imread as skimread
from skimage.color import rgb2gray as sk_rgb2gray
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import gzip, pickle
"""
UTILS FUNCTIONS
"""

def in_feature_size (ft_extrctor_p, img_size):
    w = img_size
    for lyr_name, lyr_func in ft_extrctor_p.items():
        for oprt in lyr_func.keys():
            prmtrs = lyr_func[oprt]

            if oprt == 'conv':
                num_filters = prmtrs[1]
                w = int ((w - prmtrs[2] + 2*prmtrs[4]) /prmtrs[3])+1

            elif oprt =='maxpool':
                w = int((w - prmtrs[0] + 2 * prmtrs[2]) / prmtrs[1])+1
    print (w,w,num_filters)
    return w*w*num_filters

class USPS(Dataset):
    """USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data *= 255.0
        self.train_data = (self.train_data.transpose(
            (0, 2, 3, 1))).reshape(-1,28,28).astype('uint8')  # convert to HWC

        print( ' size ' + str(self.train_data.shape) + ' input interval ' + str(
            [np.min(self.train_data), np.max(self.train_data)]))


    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index], self.train_labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label = int(label)
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels

class Local_Dataset_digit(Dataset):
    def __init__(self, data_name, set,  data_path, transform, num_samples=None, slct_trgt=None, num_chann=1):
        super(Local_Dataset_digit, self).__init__()
        self.data_path = data_path
        self.data_name = data_name
        self.set = set
        self.transform = transform
        self.num_samples = num_samples
        self.slct_trgt = slct_trgt
        self.num_chann = num_chann


        if self.data_name == 'usps':

            self.inputs , self.labels =  self._USPS()
            self.inputs, self.labels = self._select_data()
            if self.num_chann!=1:
                x = self.inputs.unsqueeze(3)
                self.inputs = torch.cat((x, x, x), 3)


        elif self.data_name=='m-mnist' or self.data_name =='mnist' :
            if self.set =='train' or self.set=='validation':
                self.inputs, self.labels = torch.load(open(self.data_path+'/processed/training.pt','rb'))



            elif self.set == 'test':
                self.inputs, self.labels = torch.load(open(self.data_path+'/processed/test.pt','rb'))


            self.inputs, self.labels = self._select_data()

            if self.data_name== 'm-mnist':

                self.inputs , self.labels = self._create_m_mnist( self.inputs,self.labels)

            elif self.data_name=='mnist' and self.num_chann!=1:
                x = self.inputs.unsqueeze(3)
                self.inputs = torch.cat((x,x,x),3)

        if self.slct_trgt is not None:
            selct_trgt_indx =  np.where(self.labels==slct_trgt)[0]

            self.inputs= self.inputs[selct_trgt_indx[:200]]
            self.labels = self.labels[selct_trgt_indx[:200]]



        print(self.data_name+' size '+str(self.inputs.size())+' input interval '+str([torch.min(self.inputs),torch.max(self.inputs)]))
        self.length = len(self.inputs)

    def __getitem__(self, index):

        img = self.inputs[index]
        lbl = int(self.labels[index])
        # img from tensor converted to numpy for applying a transformation:
        img = Image.fromarray(img.numpy())


        if self.transform is not None:
            # convert back to tensor will be done as transform
            img = self.transform(img)

        return img, lbl

    def __len__(self):

        return len(self.inputs)



    def _select_data (self):
        if self.num_samples is not None:
            classes = np.unique(self.labels)
            try :
                inputs , targets = self.inputs.numpy(), self.labels.numpy()
            except AttributeError:
                inputs, targets = self.inputs, self.labels
                pass
            if len(
                inputs) <= self.num_samples:
                print ("! requested number of samples {:d} exceed the available data {:d}!! The maximum number to request is {:d}".format(self.num_samples, len(inputs), len(inputs)))
                return self.inputs, self.labels
            s_inputs, s_targets = [],[]
            for i in (classes):
                indx = np.where((targets).astype('uint8')==i)[0]

                if self.set == 'validation':
                    s_inputs.append(inputs[indx][-100:])
                    s_targets.append(targets[indx][-100:])
                elif self.set=='train' or self.set=='test':
                    c_indx = np.random.choice(len(indx), int(self.num_samples/len(classes)),replace=False)
                    s_inputs.append(inputs[indx[c_indx]])
                    s_targets.append(targets[indx[c_indx]])

            s_inputs = np.concatenate(s_inputs, axis=0)
            s_targets = np.concatenate(s_targets, axis=0)
            s_inputs = torch.tensor(s_inputs)
            s_targets = torch.tensor(s_targets, dtype=torch.long)
            return s_inputs,s_targets

        else: return self.inputs,self.labels

    def _create_m_mnist(self,imgs,lbls):
        imgs, lbls = imgs.numpy(), lbls.numpy()
        print ('----> m_mnist'+str(imgs.shape))
        assert  len(imgs) == len(lbls)

        def _compose_image(digit, background):
            """Difference-blend a digit and a random patch from a background image."""

            w, h, _ = background.shape
            dw, dh, _ = digit.shape
            x = np.random.randint(0, w - dw)
            y = np.random.randint(0, h - dh)

            bg = background[x:x + dw, y:y + dh]
            return np.abs(bg - digit).astype(np.uint8)

        def _mnist_to_img(x):
            """Binarize MNIST digit and convert to RGB."""
            x = (x > 0).astype(np.float32)
            d = x.reshape([28, 28, 1]) * 255
            return np.concatenate([d, d, d], 2)

        def _create_mnistm(X, background_data):
            """
            Give an array of MNIST digits, blend random background patches to
            build the MNIST-M dataset as described in
            http://jmlr.org/papers/volume17/15-239/15-239.pdf
            """
            rand = np.random.RandomState(42)
            X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
            for i in range(X.shape[0]):

                bg_img = rand.choice(background_data)
                while bg_img is None: bg_img = rand.choice(background_data)
                d = _mnist_to_img(X[i])
                d = _compose_image(d, bg_img)
                X_[i] = d

            return X_

        # # import pdb ;pdb.set_trace()
        # if  not os.path.isfile(self.data_path+'/mnist_m_data.pt'):

        BST_PATH = self.data_path+'/BSR_bsds500.tgz'

        if 'BSR_bsds500.tgz' not in os.listdir(self.data_path):
            urllib.request.urlretrieve('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz',self.data_path+'/BSR_bsds500.tgz')

        f = tarfile.open(BST_PATH)
        train_files = []
        for name in f.getnames():
            if self.set=='train' or self.set=='validation':
                the_set = 'train'
            else: the_set='test'
            if name.startswith('BSR/BSDS500/data/images/'+the_set+'/'):
                    train_files.append(name)


        background_data = []
        for name in train_files:
            try:
                fp = f.extractfile(name)
                bg_img = Image.open(fp)
                background_data.append(np.array(bg_img))
            except:
                continue

        # os.remove(self.data_path+'/BSR_bsds500.tgz')

        train = _create_mnistm(imgs, background_data)
        if self.num_chann ==1:
            train = np.mean(train, axis=3)
            train = train.reshape(-1, 28, 28)
        # else:     train = train.transpose(0,3,1,2)

        train = train.astype('uint8')

        # if self.set=='train':train, lbls = self._select_data(train, lbls)
        train = torch.tensor(train)
        # lbls = torch.tensor(lbls, dtype=torch.long)

        return train, lbls

    def _USPS(self):
        def resize_and_scale(img, size):
            img = skresize(img, size)
            img = img * 255
            return 255 - (np.array(img))

        # if  os.path.isfile(self.data_path+'/USPS'+set+'.pt'):
        sz = (28, 28)
        imgs_usps = []
        lbl_usps = []
        if 'USPdata.zip' not in os.listdir(self.data_path): urllib.request.urlretrieve(
            'https://github.com/darshanbagul/USPS_Digit_Classification/raw/master/USPSdata/USPSdata.zip',
            self.data_path + '/USPSdata.zip')
        zip_ref = zipfile.ZipFile(self.data_path + '/USPSdata.zip', 'r')
        zip_ref.extractall(self.data_path)
        zip_ref.close()
        if self.set == 'train' or self.set == 'validation':
            for i in range(10):
                label_data = self.data_path + '/Numerals/' + str(i) + '/'
                img_list = os.listdir(label_data)
                for name in img_list:
                    if '.png' in name:
                        img = skimread(label_data + name)
                        img = sk_rgb2gray(img)
                        resized_img = resize_and_scale(img, sz)
                        imgs_usps.append(resized_img)
                        lbl_usps.append(i)

        elif self.set == 'test':
            test_path = self.data_path + '/Test/'
            strt = 1
            for lbl, cntr in enumerate(range(151, 1651, 150)):

                for i in range(strt, cntr):
                    i = format(i, '04d')
                    img = skimread(os.path.join(test_path, 'test_' + str(i) + '.png'))
                    img = sk_rgb2gray(img)
                    resized_img = resize_and_scale(img, sz)
                    imgs_usps.append(resized_img)
                    lbl_usps.append(9 - lbl)
                strt = cntr

        # os.remove(self.data_path+'/USPSdata.zip')
        shutil.rmtree(self.data_path + '/Numerals')
        shutil.rmtree(self.data_path + '/Test')

        imgs_usps, lbl_usps = np.asarray(imgs_usps).reshape(-1, 28, 28), np.asarray(lbl_usps)
        imgs_usps = imgs_usps.astype('uint8')
        lbl_usps = torch.tensor(lbl_usps, dtype=torch.long)
        imgs_usps = torch.tensor(imgs_usps).type(torch.torch.uint8)
        print(imgs_usps.shape, torch.max(imgs_usps))

        # torch.save((imgs_usps,lbl_usps), open(self.data_path+'/USPS'+set+'.pt','wb'))
        #
        # else:
        #     imgs_usps, lbl_usps = torch.load(open(self.data_path+'/USPS'+set+'.pt','rb'))

        return imgs_usps, lbl_usps





class Local_SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, num_smpl=None, num_chanl=1, slct_trgt=None):
        
        super(Local_SVHN, self).__init__(
            root, split, transform, target_transform, download)

        # print(' size ' + str(self.data.shape) + ' input interval ' + str([np.min(self.data), np.max(self.data)]))
        self.num_samples = num_smpl
        self.num_chanl = num_chanl
        self.data, self.labels = self._select_data()

        if slct_trgt is not None:

            # self.labels = torch.tensor(self.labels)
            selct_trgt_indx = np.where(self.labels==slct_trgt)[0]

            # selct_trgt_indx =  torch.flatten(torch.nonzero(self.labels.eq(slct_trgt)))

            self.data= self.data[selct_trgt_indx[:200]]
            self.labels = self.labels[selct_trgt_indx[:200]]

        self.labels = torch.tensor(self.labels)
        self.__repr__()

    def _select_data(self):
        if self.num_samples is not None:
            classes = np.unique(self.labels)

            inputs, targets = self.data, self.labels

            if len(inputs) <= self.num_samples:
                print("! requested number of samples {:d} exceed the available data {:d}!! The maximum number to request is {:d}".format(
                        self.num_samples, len(inputs), len(inputs)))
                return self.data, self.labels
            s_inputs, s_targets = [], []
            for i in (classes):

                indx = np.where((targets).astype('uint8') == i)[0]

                c_indx = np.random.choice(len(indx), int(self.num_samples / len(classes)),replace=False)
                s_inputs.append(inputs[indx[c_indx]])
                s_targets.append(targets[indx[c_indx]])

            s_inputs = np.concatenate(s_inputs, axis=0)
            s_targets = np.concatenate(s_targets, axis=0)
            s_inputs = s_inputs
            s_targets = s_targets
            return s_inputs, s_targets

        else:
            return self.data, self.labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], (self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.num_chanl==1:
         
        # Convert to grayscale
           img = img.convert('L')
        

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __repr__(self):
        print(len(self.data), np.min(self.data), np.max(self.data))
from scipy.misc import imsave
class Local_cifar10(torchvision.datasets.CIFAR10):
    def __init__(self,root, split='train',
                 transform=None, target_transform=None, download=False, num_smpl=None, slct_trgt=None, num_chanl= 3):
        train = split == 'train'
        super(Local_cifar10, self).__init__(root,train,transform,target_transform, download)
        self.num_chanl = num_chanl
        self.num_samples = num_smpl

        if train:

            self.data = np.asarray(self.train_data)
            self.labels = np.asarray(self.train_labels)

        else:

            self.data =np.asarray( self.test_data)
            self.labels = np.asarray(self.test_labels)

        self.data, self.labels = self._select_data()



        if slct_trgt is not None:
            # self.labels = torch.tensor(self.labels)
            selct_trgt_indx = np.where(self.labels==slct_trgt)[0]

            # selct_trgt_indx =  torch.flatten(torch.nonzero(self.labels.eq(slct_trgt)))
            print (selct_trgt_indx.shape)
            self.data= self.data[selct_trgt_indx[:500]]
            self.labels = self.labels[selct_trgt_indx[:500]]

        print(' size ' + str(self.data.shape) + ' input interval ' + str(
            [np.min(self.data), np.max(self.data)]))

        self.labels = torch.tensor(self.labels)

    def _select_data(self):
        if self.num_samples is not None:
            classes = np.unique(self.labels)

            inputs, targets = self.data, self.labels

            if len(inputs) <= self.num_samples:
                print(
                    "! requested number of samples {:d} exceed the available data {:d}!! The maximum number to request is {:d}".format(
                        self.num_samples, len(inputs), len(inputs)))
                return self.data, self.labels
            s_inputs, s_targets = [], []
            for i in (classes):
                indx = np.where((targets).astype('uint8') == i)[0]
                c_indx = np.random.choice(len(indx), int(self.num_samples / len(classes)), replace=False)
                s_inputs.append(inputs[indx[c_indx]])
                s_targets.append(targets[indx[c_indx]])

            s_inputs = np.concatenate(s_inputs, axis=0)
            s_targets = np.concatenate(s_targets, axis=0)

            return s_inputs, s_targets

        else:
            return self.data, self.labels


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], (self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.num_chanl == 1:
            # Convert to grayscale
            img = img.convert('L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
#        print (img.size(),target.size())
        return img, target

    def __len__(self):
        return len(self.data)



class Local_cifar100(torchvision.datasets.CIFAR100):
    def __init__(self,root, split='train',
                 transform=None, target_transform=None, download=False, num_smpl=None, slct_trgt=None, num_chanl= 3):
        train = split == 'train'
        super(Local_cifar100, self).__init__(root,train,transform,target_transform, download)
        self.num_chanl = num_chanl
        self.num_samples = num_smpl
        if train:

            self.data = np.asarray(self.train_data)
            self.labels = np.asarray(self.train_labels)
        else:


            self.data =np.asarray( self.test_data)
            self.labels = np.asarray(self.test_labels)

        self.data, self.labels = self._select_data()


        if slct_trgt is not None:
            # self.labels = torch.tensor(self.labels)
            selct_trgt_indx = np.where(self.labels==slct_trgt)[0]

            # selct_trgt_indx =  torch.flatten(torch.nonzero(self.labels.eq(slct_trgt)))
            print (selct_trgt_indx.shape)
            self.data= self.data[selct_trgt_indx[:500]]
            self.labels = self.labels[selct_trgt_indx[:500]]



        print(' size ' + str(self.data.shape) + ' input interval ' + str(
            [np.min(self.data), np.max(self.data)]))
        self.labels = torch.tensor(self.labels)


    def _select_data(self):
        if self.num_samples is not None:
            classes = np.unique(self.labels)

            inputs, targets = self.data, self.labels

            if len(inputs) <= self.num_samples:
                print(
                    "! requested number of samples {:d} exceed the available data {:d}!! The maximum number to request is {:d}".format(
                        self.num_samples, len(inputs), len(inputs)))
                return self.data, self.labels
            s_inputs, s_targets = [], []
            for i in (classes):
                indx = np.where((targets).astype('uint8') == i)[0]
                c_indx = np.random.choice(len(indx), int(self.num_samples / len(classes)),replace=False)
                s_inputs.append(inputs[indx[c_indx]])
                s_targets.append(targets[indx[c_indx]])

            s_inputs = np.concatenate(s_inputs, axis=0)
            s_targets = np.concatenate(s_targets, axis=0)
            # s_inputs = torch.tensor(s_inputs)
            # s_targets = torch.tensor(s_targets, dtype=torch.long)
            return s_inputs, s_targets

        else:
            return self.data, self.labels
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], (self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.num_chanl == 1:
            # Convert to grayscale
            img = img.convert('L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
#        print (img.size(),target.size())
        return img, target

    def __len__(self):
        return len(self.data)

#Include_label = [1,2,3,4,5,6,9,10,14,13,15,17]
class Local_CIFAR100_noveralp(torchvision.datasets.CIFAR100):
    def __init__(self,root, split='train',
                 transform=None, target_transform=None, download=False, num_smpl=None):
        self.train = split == 'train'
        self.num_samples = num_smpl
        self.transform = transform

        super(Local_CIFAR100_noveralp, self).__init__(root,self.train,transform,target_transform, download)
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.labels = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])

                self.labels.extend(entry['coarse_labels'])


        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.labels= np.hstack(self.labels)

        Include_label = [1,2,3,4,5,6,9,10,14,13,15,17]
        X_train = []
        y_train = []
        for label in Include_label:
            indx = np.where(self.labels== label)[0]
            X_train.append(self.data[indx])
            y_train.append(10+self.labels[indx]*0)

        self.data = np.vstack(X_train).astype('uint8')
        self.labels= np.hstack(y_train).astype('int')
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.data, self.labels = self._select_data()
        self.labels = torch.tensor(self.labels)

        self.__repr__()


    def _select_data(self):
        if self.num_samples is not None:
            classes = np.unique(self.labels)

            inputs, targets = self.data, self.labels

            if len(inputs) <= self.num_samples:
                print(
                    "! requested number of samples {:d} exceed the available data {:d}!! The maximum number to request is {:d}".format(
                        self.num_samples, len(inputs), len(inputs)))
                return self.data, self.labels
            s_inputs, s_targets = [], []
            for i in (classes):
                indx = np.where((targets).astype('uint8') == i)[0]
                c_indx = np.random.choice(len(indx), int(self.num_samples / len(classes)),replace=False)
                s_inputs.append(inputs[indx[c_indx]])
                s_targets.append(targets[indx[c_indx]])

            s_inputs = np.concatenate(s_inputs, axis=0)
            s_targets = np.concatenate(s_targets, axis=0)
            # s_inputs = torch.tensor(s_inputs)
            # s_targets = torch.tensor(s_targets, dtype=torch.long)
            return s_inputs, s_targets

        else:
            return self.data, self.labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], (self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        print(' size ' + str(self.data.shape) + ' input interval ' + str(
            [np.min(self.data), np.max(self.data)]) + str((self.labels).shape))

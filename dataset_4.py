import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation

def normalize(data):
    return data/255.

def Im2Patch(img_A, win, stride=1):
    k = 0
    endc = img_A.shape[0]
    endw = img_A.shape[1]
    endh = img_A.shape[2]
    patch = img_A[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img_A[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(data_path_A, data_path_B, data_path_val_A, data_path_val_B, patch_size, stride, aug_times=1, if_reseize=False):
    # train
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]
    files_A = glob.glob(os.path.join('datasets', data_path_A, '*.*'))
    files_A.sort()

    files_B = glob.glob(os.path.join('datasets', data_path_B, '*.*'))
    files_B.sort()


    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in range(len(files_A)):
        img_A = cv2.imread(files_A[i])
        img_B = cv2.imread(files_B[i])
        
        h, w, c = img_A.shape
        if if_reseize == True:
            img_B = cv2.resize(img_B, (h, w), interpolation=cv2.INTER_CUBIC)

        #h, w, c = img_B.shape
        for k in range(len(scales)):
            Img_A = cv2.resize(img_A, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img_A = np.expand_dims(Img_A[:,:,0].copy(), 0)
            Img_A = np.float32(normalize(Img_A))

            Img_B = cv2.resize(img_B, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img_B = np.expand_dims(Img_B[:,:,0].copy(), 0)
            Img_B = np.float32(normalize(Img_B))

            patches_A = Im2Patch(Img_A, win=patch_size, stride=stride)
            patches_B = Im2Patch(Img_B, win=patch_size, stride=stride)

            print("file: %s scale %.1f # samples: %d" % (files_A[i], scales[k], patches_A.shape[3]*aug_times))
            print("file: %s scale %.1f # samples: %d" % (files_B[i], scales[k], patches_A.shape[3]*aug_times))
            
            for n in range(patches_A.shape[3]):
                data_A = patches_A[:,:,:,n].copy()
                data_B = patches_B[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=[data_A, data_B])

                train_num += 1
                for m in range(aug_times-1):
                    rand = np.random.randint(1,8)
                    data_aug_A = data_augmentation(data_A, rand)
                    data_aug_B = data_augmentation(data_B, rand)
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=[data_aug_A, data_aug_B])
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files_A.clear()
    files_A = glob.glob(os.path.join('datasets', data_path_val_A, '*.*'))
    files_A.sort()

    files_B.clear()
    files_B = glob.glob(os.path.join('datasets', data_path_val_B, '*.*'))
    files_B.sort()

    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files_A)):
        print("file: %s" % files_A[i])
        img_A = cv2.imread(files_A[i])
        img_A = np.expand_dims(img_A[:,:,0], 0)
        img_A = np.float32(normalize(img_A))

        img_B = cv2.imread(files_B[i])
        if if_reseize == True:
            h, w, c = img_B.shape
            img_B = cv2.resize(img_B, (h*4, w*4), interpolation=cv2.INTER_CUBIC)
        img_B = np.expand_dims(img_B[:,:,0], 0)
        img_B = np.float32(normalize(img_B))

        h5f.create_dataset(str(val_num), data=[img_A,img_B])
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)

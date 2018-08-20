import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.*'))
    filesn_source = glob.glob(os.path.join('noise', opt.test_data, '*.*'))

    files_source.sort()
    filesn_source.sort()
    # process data
    psnr_test = 0
    psnrn_test = 0
    for f in range(len(files_source)):
        # image
        Img = cv2.imread(files_source[f])
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        #noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        #INoisy = ISource + noise
        Img_N = cv2.imread(filesn_source[f])
        Img_N = normalize(np.float32(Img_N[:,:,0]))
        Img_N = np.expand_dims(Img_N, 0)
        Img_N = np.expand_dims(Img_N, 1)
        INoisy = torch.Tensor(Img_N)
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        psnrn = batch_PSNR(INoisy, ISource, 1.)
        psnrn_test += psnrn
        print("%s PSNR %f" % (f, psnr))
        print("%s PSNRn %f" % (f, psnrn))
        Out= Out[0,:,:].cpu()
        Out= Out[0].numpy().astype(np.float32)*255
        cv2.imwrite("/home/richard/Documents/DnCNN-PyTorch-3/small/%#04d.png" % (f+3001), Out)

    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    psnrn_test /= len(files_source)
    print("\nPSNRn on test data %f" % psnrn_test)
    

    print(type( Img )) #
    ISource = ISource[0,:,:].cpu()
    ISource = ISource[0].numpy().astype(np.float32)
    INoisy= INoisy[0,:,:].cpu()
    INoisy= INoisy[0].numpy().astype(np.float32)
    

    print ('aa')
    print(type( Img )) #
    fig = plt.figure()

    ax = plt.subplot("131")
    ax.imshow(ISource, cmap='gray') #
    ax.set_title("GT")

    ax = plt.subplot("132")
    ax.imshow(INoisy, cmap='gray') #, cmap='gray'
    ax.set_title("Input(with 'realistic' noise)")

    ax = plt.subplot("133")
    ax.imshow(Out, cmap='gray')
    ax.set_title("Output(DnCNN)")
    plt.show()
    print ('a')

if __name__ == "__main__":
    main()

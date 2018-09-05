import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset_AB import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--A", type=str, default="train_S_A", help='path of files targeted')
parser.add_argument("--B", type=str, default="train_S_B", help='path of files to process')
parser.add_argument("--val_A", type=str, default="val_S_A", help='path of files targeted')
parser.add_argument("--val_B", type=str, default="val_S_B", help='path of files to process')

parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='Super-resolution (S) or denoise training (N)')

opt = parser.parse_args()

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_A_train = data[:,0]
            img_B_train = data[:,1,:,:]

            difference = img_B_train - img_A_train
            print(difference.size())

            img_A_train, img_B_train = Variable(img_A_train.cuda()), Variable(img_B_train.cuda())
            difference = Variable(difference.cuda())
            out_train = model(img_B_train)
            loss = criterion(out_train, difference) / (img_B_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train = torch.clamp(img_B_train-model(img_B_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_A_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            torch.save(model.state_dict(), os.path.join(opt.outf,"epoch_%d_net.pth" %(epoch+1)))
        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0
        
        for k in range(len(dataset_val)):
            img_val_A = torch.unsqueeze(dataset_val[k][0], 0)
            imgn_val_B = torch.unsqueeze(dataset_val[k][1], 0)
            difference = imgn_val_B - img_val_A
            img_val_A, imgn_val_B = Variable(img_val_A.cuda()), Variable(imgn_val_B.cuda())
            out_val = torch.clamp(model(imgn_val_B), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val_A, 1.)
        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        
        # log the images
        out_train = torch.clamp(img_B_train-model(img_B_train), 0., 1.)
        Img_A = utils.make_grid(img_A_train.data, nrow=8, normalize=True, scale_each=True)
        Img_B = utils.make_grid(img_B_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img_A, epoch)
        writer.add_image('input image', Img_B, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path_A=opt.A, data_path_B=opt.B, data_path_val_A=opt.val_A, data_path_val_B=opt.val_B,  patch_size=50, stride=10, aug_times=1, if_reseize=True)
        if opt.mode == 'N':
            prepare_data(data_path_A=opt.A, data_path_B=opt.B, data_path_val_A=opt.val_A, data_path_val_B=opt.val_B,  patch_size=30, stride=10, aug_times=2, if_reseize=False)
    main()

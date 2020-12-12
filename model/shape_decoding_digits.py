'''
Training contour decoders
Reconstruct shapes of stimulus images from the fMRI inputs
'''
import argparse
import os
import numpy as np
import math
import itertools
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch
import torchvision
import matplotlib.pyplot as plt
import scipy

np.random.seed(0)

def rasterize_img(img, out_size):
    raw_sz = img.shape[1]
    per_check = raw_sz/out_size
    new_arr = np.zeros((1, out_size, out_size))
    for l in range(out_size):
        for p in range(out_size):
            sx = int(l*per_check)
            ex = int((l+1)*per_check)
            sy = int(p*per_check)
            ey = int((p+1)*per_check)
            check = img[0, sx:ex, sy:ey]
            sum = np.sum(np.reshape(check, (check.size,)))
            avg = sum*1.0 / (per_check*per_check)
            new_arr[0, l, p] = avg
    return new_arr


def de_rasterize_img(img, out_size):
    raw_sz = img.shape[1]
    times = out_size/raw_sz
    new_arr = np.zeros((1, out_size, out_size))
    for l in range(raw_sz):
        for p in range(raw_sz):
            sx = int(l*times)
            ex = int((l+1)*times)
            sy = int(p*times)
            ey = int((p+1)*times)
            new_arr[0, sx:ex, sy:ey] = img[0, l, p]
    return new_arr


class Decoder(nn.Module):
    def __init__(self, fmri_size, latent_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(fmri_size, latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        return x


def save_demo1_images(sp_imgs, raw_imgs):
    for i in range(90, 100):
        plt.figure()
        im = sp_imgs[i]
        im = Image.fromarray(im)
        im = im.convert('RGB')

        raw_img = Image.fromarray(raw_imgs[i])
        raw_img = raw_img.convert('RGB')
        target = Image.new('RGB', (28 * 2, 28))
        target.paste(raw_img, (0, 0, 28, 28))
        target.paste(im, (28, 0, 56, 28))
        target = target.resize((512, 256), Image.ANTIALIAS)
        target = target.convert('L')
        target = np.asarray(target)
        target = target[:, ::-1]

        ax = plt.subplot(1, 1, 1)
        plt.imshow(np.fliplr(target), cmap='hot')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig("results/digits/shape_decoding%d.png" % i)


def decoding_shapes(train_fmri, train_imgs, test_fmri, test_imgs, output_size, n_epochs, batch_size,
                    lr, b1=0.5, b2=0.999):

    latent_dim = output_size*output_size
    rand_id = np.random.randint(low=0, high=train_fmri.shape[0], size=train_fmri.shape[0])
    train_fmri = train_fmri[rand_id]
    fmri = np.concatenate([train_fmri, test_fmri])


    train_imgs = train_imgs[rand_id]
    imgs = np.concatenate([train_imgs, test_imgs])
    raw_imgs = imgs

    total_blocks = fmri.shape[0]
    fmri_size = fmri.shape[1]

    train_num = total_blocks - 10

    batch_num = train_num / batch_size

    print('Train blocks:'+str(train_num))
    print('batch num:' + str(batch_num))
    cuda = True if torch.cuda.is_available() else False


    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss()
    # Initialize generator and discriminator
    decoder = Decoder(fmri_size=fmri_size, latent_dim=latent_dim)

    if cuda:
        decoder.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()


    optimizer_E = torch.optim.Adam(decoder.parameters(), lr=lr, betas=(b1, b2))
    scheduler_E = torch.optim.lr_scheduler.StepLR(optimizer_E, step_size=5, gamma=0.7, last_epoch=-1)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    print(fmri.shape)
    fmri = torch.from_numpy(fmri)
    fmri = fmri.type(Tensor)


    imgs = torch.from_numpy(imgs)
    imgs = imgs.type(Tensor)


    for epoch in range(n_epochs + 1):
        for i in range(0, int(batch_num)):
            fmri_data = fmri[i * batch_size:(i + 1) * batch_size]
            real_imgs = imgs[i * batch_size:(i + 1) * batch_size]

            optimizer_E.zero_grad()

            latent_vector = decoder(fmri_data)

            obj_vector = real_imgs.reshape(real_imgs.shape[0], -1)

            e_loss = pixelwise_loss(obj_vector, latent_vector)

            e_loss.backward()
            optimizer_E.step()

            print(
                "[Epoch %d/%d] [Batch %d] [E loss: %f]"
                % (epoch, n_epochs, i, e_loss.item())
            )

        if epoch % 10 == 0:
            test_fmri_data = fmri[train_num:train_num + batch_size]
            latent_v = decoder(test_fmri_data)
            latent_v = latent_v.view(batch_size, 1, output_size, output_size)

            tempv = latent_v.data

    imgs = decoder(fmri)
    imgs = imgs.view(fmri.shape[0], output_size, output_size)
    imgs = imgs.data.cpu() * 255.0
    imgs = np.asarray(imgs)

    # save_demo1_images(imgs, raw_imgs)
    return decoder, imgs, raw_imgs

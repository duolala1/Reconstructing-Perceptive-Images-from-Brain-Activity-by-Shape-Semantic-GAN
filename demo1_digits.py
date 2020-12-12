'''
The codes of paper:
"Reconstructing Perceptive Images from Brain Activity by Shape-Semantic GAN".
This demo takes the published fMRI data "69-digits" as inputs.

The shape decoding and semantic decoding scripts could be found in "model" fold.

Tao Fang, 2020.12
'''
import torch.nn as nn
from model import shape_decoding_digits
import matplotlib.pyplot as plt
import time
import datetime
from torch.utils.data import DataLoader
from model.models import *
from data.datasets import *
import torch


fmri = torch.load('data/fMRI_data/demo1/digits-fmri')
imgs = torch.load('data/images/demo1/raw_imgs/digits-images')

train_fmri = np.concatenate([fmri[0:45], fmri[50:95]])
test_fmri = np.concatenate([fmri[45:50], fmri[95:100]])

train_imgs = np.concatenate([imgs[0:45], imgs[50:95]])
test_imgs = np.concatenate([imgs[45:50], imgs[95:100]])

print('Training shape decoder:')
sp_decoder, sp_imgs, raw_imgs = shape_decoding_digits.decoding_shapes(train_fmri, train_imgs, test_fmri, test_imgs, output_size=28, n_epochs=101,
                                           batch_size=10, lr=0.0006)

for i in range(0, 90):
    im = sp_imgs[i]
    im = Image.fromarray(im)
    im = im.convert('RGB')

    raw_img = Image.fromarray(raw_imgs[i])
    raw_img = raw_img.convert('RGB')
    target = Image.new('RGB', (28 * 2, 28))
    target.paste(raw_img, (0, 0, 28, 28))
    target.paste(im, (28, 0, 56, 28))
    target = target.resize((512, 256), Image.ANTIALIAS)
    target.save('data/images/demo1/samples/train/%d.jpg' % i)

for i in range(90, 100):
        im = sp_imgs[i]
        im = Image.fromarray(im)
        im = im.convert('RGB')

        raw_img = Image.fromarray(raw_imgs[i])
        raw_img = raw_img.convert('RGB')
        target = Image.new('RGB', (28 * 2, 28))
        target.paste(raw_img, (0, 0, 28, 28))
        target.paste(im, (28, 0, 56, 28))
        target = target.resize((512, 256), Image.ANTIALIAS)
        target.save('data/images/demo1/samples/val/%d.jpg' % i)


n_epochs = 10
batch_size = 10
lr = 0.0002
b1 = 0.5
b2 = 0.999
img_height = 256
img_width = 256
channels = 3
dataset_name = '69digits'
interval = 2
test_sample_num = 10


cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.MSELoss()
criteirion_contour = torch.nn.MSELoss()
# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 150

# Calculate output of image discriminator (PatchGAN)
patch = (1, img_height // 2 ** 4, img_width // 2 ** 4)


# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()


if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    criteirion_contour.cuda()


# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=3, gamma=0.9)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=3, gamma=0.9)


# Configure dataloaders
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# training set
dataloader = DataLoader(
    ImageDataset("data/images/demo1/samples", transforms_=transforms_, mode='train'),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

# validation set
val_dataloader = DataLoader(
    ImageDataset("data/images/demo1/samples", transforms_=transforms_, mode="val"),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)


# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print('Load semantic features:')
semantic_vecs = torch.load('data/sm_features/demo1/semantics')

train_semantic_vec = semantic_vecs[0:90]
test_semantic_vec = semantic_vecs[90:100]

train_semantic_vec = train_semantic_vec.type(Tensor)
test_semantic_vec = test_semantic_vec.type(Tensor)
prev_time = time.time()

for epoch in range(0, n_epochs):
    scheduler_D.step()
    scheduler_G.step()
    for i, batch in enumerate(dataloader):
        # Model inputs
        real_A = batch["B"].type(Tensor)   # 轮廓
        real_B = batch["A"].type(Tensor)   # 真图
        # Adversarial ground truths
        valid = Tensor(np.ones((real_A.size(0), *patch)))
        fake = Tensor(np.zeros((real_A.size(0), *patch)))

        #  Train Generator

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A, train_semantic_vec[i*batch_size:(i+1)*batch_size])

        pred_fake = discriminator(fake_B, real_A)

        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        #  Train Discriminator

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        #  Log Progress
        batches_done = epoch * len(dataloader) + i
        batches_left = n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
        prev_time = time.time()

        print(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

k = 0
for j, batch in enumerate(val_dataloader):
        plt.figure()
        real_A = batch["B"].type(Tensor)
        real_B = batch["A"].type(Tensor)
        fake_B = generator(real_A, test_semantic_vec)

        fake_B = fake_B.cpu().data
        fake_B = np.asarray(fake_B)
        real_B = real_B.cpu().data
        real_B = np.asarray(real_B)

        plt.figure()
        for i in range(10):
            fimg = fake_B[i]
            fimg = 0.3 * fimg[0]+0.59 * fimg[1]+0.11*fimg[2]
            fimg[fimg < 0] = 0
            fimg = fimg[:, ::-1]
            fimg = fimg.reshape(256, 256) * 255


            rimg = real_B[i]
            rimg = 0.3 * rimg[0]+0.59 * rimg[1]+0.11*rimg[2]
            rimg[rimg < 0] = 0
            rimg = rimg[:, ::-1]
            rimg = rimg.reshape(256, 256) * 255

            im = fimg
            im = Image.fromarray(im)
            im = im.convert('RGB')

            raw_img = Image.fromarray(rimg)
            raw_img = raw_img.convert('RGB')
            target = Image.new('RGB', (256 * 2, 256))
            target.paste(raw_img, (0, 0, 256, 256))
            target.paste(im, (256, 0, 512, 256))
            target = target.resize((512, 256), Image.ANTIALIAS)
            target = target.convert('L')
            target = np.asarray(target)

            ax = plt.subplot(5, 2, i+1)
            plt.imshow(np.fliplr(target), cmap='hot')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig("results/digits/reconstructed_img.png")

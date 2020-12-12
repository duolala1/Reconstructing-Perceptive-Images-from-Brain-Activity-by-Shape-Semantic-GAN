import numpy as np
import argparse
import os
from sklearn import preprocessing
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision
import matplotlib.pyplot as plt

n_epochs=100
batch_size = 10
lr = 0.0002
b1 = 0.5
b2 = 0.999

np.random.seed(0)
fmri = torch.load('../data/fMRI_data/demo1/digits-fmri')
raw_labels = torch.load('../data/images/demo1/raw_imgs/digits-labels') - 1
labels = raw_labels

newlabels = labels
print(fmri.shape)
print(labels.shape)

total_blocks = fmri.shape[0]
fmri_size = fmri.shape[1]
print('total blocks:'+str(total_blocks))
print('input fmri size:'+str(fmri_size))

test_num = 10

cuda = True if torch.cuda.is_available() else False


class Decoder(nn.Module):
    def __init__(self, fmri_size):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(fmri_size, 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
        )

        self.classify = nn.Sequential(
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, fmri):
        fmri_flat = fmri.view(fmri.shape[0], -1)
        semantics = self.model(fmri_flat)
        x = self.classify(semantics)
        return x, semantics

# Use binary cross-entropy loss
adversarial_loss = torch.nn.CrossEntropyLoss()
encoder = Decoder(fmri_size=fmri_size)

if cuda:
    encoder.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(b1, b2))
scheduler_E = torch.optim.lr_scheduler.StepLR(optimizer_E, step_size=10, gamma=0.8, last_epoch=-1)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

train_fmri = np.concatenate([fmri[0:45], fmri[50:95]])
test_fmri = np.concatenate([fmri[45:50], fmri[95:100]])

rand_id = np.random.randint(low=0, high=train_fmri.shape[0], size=train_fmri.shape[0])
train_fmri = train_fmri[rand_id]
fmri = np.concatenate([train_fmri, test_fmri])
print(fmri.shape)


train_labels = np.concatenate([labels[0:45], labels[50:95]])
test_labels = np.concatenate([labels[45:50], labels[95:100]])
train_labels = train_labels[rand_id]
labels = np.concatenate([train_labels, test_labels])

fmri = torch.from_numpy(fmri)
fmri = fmri.type(Tensor)

train_fmri = torch.from_numpy(train_fmri)
train_labels = torch.from_numpy(train_labels)
train_labels = train_labels.squeeze()
train_fmri = train_fmri.type(Tensor)
train_labels = train_labels.type(Tensor)

test_fmri = torch.from_numpy(test_fmri)
test_labels = torch.from_numpy(test_labels)
test_labels = test_labels.squeeze()
test_fmri = test_fmri.type(Tensor)
test_labels = test_labels.type(Tensor)

acc_vec = []
train_vec = []
loss_vec = []
for epoch in range(n_epochs):
    for i in range(0, 9):
            fmri_data = train_fmri[i*batch_size:(i+1) * batch_size]
            labels_data = train_labels[i*batch_size:(i+1) * batch_size]
            labels_data = labels_data.long()
            optimizer_E.zero_grad()
            predict, _ = encoder(fmri_data)
            e_loss = adversarial_loss(predict, labels_data)
            e_loss.backward()
            optimizer_E.step()
            loss_vec.append(e_loss.item())
            print(
                "[Epoch %d/%d] [Batch %d] [loss: %f] "
                % (epoch, n_epochs, i, e_loss.item())
            )

    if epoch % 1 == 0:
            test_fmri_data = test_fmri
            test_label_data = test_labels.cpu().detach().numpy()
            lbs, _ = encoder(test_fmri_data)
            cpu_labels = lbs.cpu().detach().numpy()
            pred = [np.argmax(lb) for lb in cpu_labels]
            num_correct = (pred == test_label_data).sum()
            acc = num_correct / 10
            acc_vec.append(acc)


    scheduler_E.step()
    lrd = optimizer_E.param_groups[0]['lr']


plt.figure()
plt.title('Loss of classification task - 2 class digits')
plt.plot(loss_vec)
plt.xlabel('Epoch num')
plt.ylabel('Loss')
plt.show()

pred, sm = encoder(fmri)
torch.save(sm, '../data/sm_features/demo1/semantics')

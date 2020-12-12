import torch.nn as nn

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

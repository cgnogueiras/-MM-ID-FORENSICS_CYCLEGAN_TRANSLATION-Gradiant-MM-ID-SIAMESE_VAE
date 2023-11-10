import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import functional as F
from utils.losses import ContrastiveLoss


class VAE(pl.LightningModule):
    def __init__(
        self,
        latent_dim=256,
        input_width: int = 500,
        input_height: int = 200,
        input_channels=3,
    ):
        super().__init__()
        input_dim = input_channels * input_height * input_width
        self.input_dim = input_dim
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.similarity = ContrastiveLoss()

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, 1500)
        self.fc2 = nn.Linear(1500, 1204)
        self.fc3 = nn.Linear(1204, 908)
        self.fc4 = nn.Linear(908, 612)
        self.fc5 = nn.Linear(612, latent_dim)

        # distribution parameters
        self.fc_mu = nn.Linear(612, latent_dim)
        self.fc_var = nn.Linear(612, latent_dim)

        # Decoder layers
        self.fc5 = nn.Linear(latent_dim, 612)
        self.fc6 = nn.Linear(612, 908)
        self.fc7 = nn.Linear(908, 1204)
        self.fc8 = nn.Linear(1204, 1500)
        self.fc9 = nn.Linear(1500, input_dim)

        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        h4 = self.fc4(h3)
        return h4

    def decode(self, z):
        h6 = self.relu(self.fc5(z))
        h7 = self.relu(self.fc6(h6))
        h8 = self.relu(self.fc7(h7))
        h9 = self.relu(self.fc8(h8))
        return self.sigmoid(self.fc9(h9))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x1, x2):
        x1_encoded = self.encode(x1.view(x1.shape[0], -1))
        x2_encoded = self.encode(x2.view(x2.shape[0], -1))

        mu1, logvar1 = self.fc_mu(x1_encoded), self.fc_var(x1_encoded)
        mu2, logvar2 = self.fc_mu(x2_encoded), self.fc_var(x2_encoded)

        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)

        return self.decode(z1), z1, mu1, logvar1, self.decode(z2), z2, mu2, logvar2

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(
            recon_x, x.view(x.shape[0], -1), reduction="sum"
        )
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

        # Reconstruction + KL divergence losses summed over all elements and batch

    # def loss_function(self, recon_x, x, mu, logvar):
    #     # see Appendix B from VAE paper:
    #     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    #     # https://arxiv.org/abs/1312.6114
    #     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    #     # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') #.view(-1, 784)
    #     BCE = F.mse_loss(recon_x, x, size_average=False) / 100
    #     KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / 100

    #     # print("BCE", BCE)
    #     # print("KLD", KLD)

    #     return BCE + 0.01 * KLD

    def training_step(self, batch, batch_idx):
        x1, x2, target1, target2, label = batch

        x1_hat, z1, mu1, logvar1, x2_hat, z2, mu2, logvar2 = self(x1, x2)
        loss_rec1 = self.loss_function(x1_hat, target1, mu1, logvar1)
        loss_rec2 = self.loss_function(x2_hat, target2, mu2, logvar2)

        loss_reconstruction = (loss_rec1 + loss_rec2) / 2

        loss_contrastive = self.similarity(z1, z2, label.float())
        total_loss = loss_reconstruction + 0.000001 * loss_contrastive

        self.log("train_reconstruction_loss", loss_reconstruction)
        self.log("train_contrastive_loss", loss_contrastive)
        self.log("train_total_loss", total_loss)

        self.training_step_outputs.append(total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x1, x2, target1, target2, label = batch

        x1_hat, z1, mu1, logvar1, x2_hat, z2, mu2, logvar2 = self(x1, x2)
        loss_rec1 = self.loss_function(x1_hat, target1, mu1, logvar1)
        loss_rec2 = self.loss_function(x2_hat, target2, mu2, logvar2)

        loss_reconstruction = (loss_rec1 + loss_rec2) / 2
        loss_contrastive = self.similarity(z1, z2, label.float())

        total_loss = loss_reconstruction + 0.000001 * loss_contrastive

        self.log("val_reconstruction_loss", loss_reconstruction)
        self.log("val_contrastive_loss", loss_contrastive)
        self.log("val_total_loss", total_loss)

        self.validation_step_outputs.append(total_loss)
        return total_loss

    def test_step(self, batch, batch_idx):
        x1, x2, target1, target2, label = batch

        x1_hat, z1, mu1, logvar1, x2_hat, z2, mu2, logvar2 = self(x1, x2)
        loss_rec1 = self.loss_function(x1_hat, target1, mu1, logvar1)
        loss_rec2 = self.loss_function(x2_hat, target2, mu2, logvar2)

        loss_reconstruction = (loss_rec1 + loss_rec2) / 2

        loss_contrastive = self.similarity(z1, z2, label.float())
        total_loss = loss_reconstruction + 0.000001 * loss_contrastive

        self.log("test_reconstruction_loss", loss_reconstruction)
        self.log("test_contrastive_loss", loss_contrastive)
        self.log("test_total_loss", total_loss)

        return total_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer


# Initialize the model parameters with Xavier initialization
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

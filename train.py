import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import yaml
import torch
import argparse
import random

import pytorch_lightning as pl
import torchvision.transforms as TF
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


from utils.mlflow_logger import MLflowLogger
from utils.dataloader import SiameseNetworkDataset
from utils.generate_callback import GenerateCallback

from model.Siamese_VAE import VAE
from utils.vis_fn import save_reconstructed_images

mlflow_logger = MLflowLogger(
    tracking_uri="http://10.5.0.58:8999",
    experiment_name="MM-PR-MM-SiameseAE-runs",
)


def train(
    train_dataset, train_loader, val_loader, latent_dim, device, input_channels=1
):
    """Create a PyTorch Lightning trainer with the generation callback.
    Check whether pretrained model exists. If yes, load it and skip training.
    Test best model on validation and test set.

    Args:
        train_dataset (dataset): Train dataset
        train_loader (torch dataloader):  Train dataloader
        val_loader (torch dataloader): Validation split dataloader
        test_loader (torch dataloader): Test split dataloader
        latent_dim (int): Dimensionality of latent representation z
        device (torch.device): Device (cuda, cpu)

    Returns:
        model: model
        result (dic): test and validation results
    """
    x, y, z, a = [], [], [], []
    for i in range(8):
        i1, i2, t1, t2, _ = train_dataset[i]
        x.append(i1)
        y.append(t1)
        z.append(i2)
        a.append(t2)
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    z = torch.stack(z, dim=0)
    a = torch.stack(a, dim=0)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(
            hparams["cp_path"], f"{hparams['cp_name']}_{latent_dim}"
        ),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=[device.index],
        max_epochs=hparams["num_epochs"],
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback(
                mlflow_logger,
                x,
                y,
                z,
                a,
                input_channels=input_channels,
                results_path=hparams["results_path"],
                every_n_epochs=hparams["freq_stats"],
            ),
            LearningRateMonitor("epoch"),
        ],
    )
    model = VAE(
        latent_dim=hparams["latent_dim"],
        input_height=hparams["img_width"],
        input_width=hparams["img_heigh"],
        input_channels=input_channels,
    ).to(device)

    pretrained_filename = hparams["cp_path"] + "/" + hparams["cp_name"]

    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = model.load_from_checkpoint(pretrained_filename)
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)

    val_result = trainer.test(model, val_loader, verbose=True)
    result = {"val": val_result}

    return model, result


@mlflow_logger.attach
def main(hparams: dict):
    for artifact in ["hparams.yml", "model/Siamese_VAE.py"]:
        mlflow_logger.log_artifact(artifact)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    pl.seed_everything(hparams["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(hparams["device"] if torch.cuda.is_available() else "cpu")

    train_transform = TF.Compose(
        [
            TF.ToTensor(),
            TF.Normalize(
                mean=hparams["normalization_mean"], std=hparams["normalization_std"]
            ),
        ]
    )

    train_set, val_set = torch.utils.data.random_split(
        SiameseNetworkDataset(
            hparams["annot_path"],
            data_dir=hparams["data_path"],
            transform=train_transform,
        ),
        [0.8, 0.2],
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=hparams["batch_size"], num_workers=4
    )

    os.makedirs(hparams["cp_path"], exist_ok=True)
    os.makedirs(hparams["results_path"], exist_ok=True)

    in_channels = 3 if hparams["color"] == "RGB" else 1

    model, result = train(
        train_set,
        train_loader,
        val_loader,
        hparams["latent_dim"],
        device,
        input_channels=in_channels,
    )

    x, y, z, a = [], [], [], []
    for i in range(8):
        i1, i2, t1, t2, _ = train_dataset[i]
        x.append(i1)
        y.append(t1)
        z.append(i2)
        a.append(t2)
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    z = torch.stack(z, dim=0)
    a = torch.stack(a, dim=0)

    save_reconstructed_images(
        model,
        x,
        y,
        z,
        a,
        color=in_channels,
        results_path=hparams["results_path"],
        epoch="last",
        n=8,
    )

    mlflow_logger.log_artifact(hparams["results_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using hyperparameters from a YAML file."
    )
    parser.add_argument(
        "hparams", type=str, help="Path to the YAML file containing hyperparameters."
    )
    args = parser.parse_args()

    with open(args.hparams, "r") as f:
        try:
            hparams = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    main(hparams)

# %%
import os
from functools import cached_property, cache
from typing import Tuple, List
from pandas import DataFrame, read_parquet

import numpy as np
import torch
import zarr
from tifffile import imread, TiffFile
from torch.utils.data import Dataset

import pandas as pd
import datetime


from pandas import read_parquet
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
import logging
from typing import Tuple
import datetime


class CustomDataset(Dataset):
    def __init__(self, df_path: str):
        super(CustomDataset, self).__init__()
        self.df = read_parquet(df_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """General getitem function for sbrnet-related datasets. require the stack of lightfield views,
          and the refocused volume as inputs. and the the ground truth target.

        Args:
            index (int): index of the data. the input measurement data is stored in the format of meas_{index}.tiff,
            and the output is stored in the format of gt_vol_{index}.tiff. yours may change, so adjust this function accordingly.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: your data in torch tensor form normalized to [0,1] with 32bit float.
        """

        image = imread(self.df["image_path"].iloc[index]).astype(np.float32) / 0xFFFF
        image = torch.from_numpy(image)

        label = imread(self.df["label_path"].iloc[index]).astype(np.float32) / 0xFF
        label = torch.from_numpy(label)

        return image[None, ...], label[None, ...]


class ZarrData:
    def __init__(self, df: DataFrame, datatype: str):
        self.df = df
        self.datatype = datatype

    # NOTE: ensure cache is larger than number of items
    @cache
    def __getitem__(self, index: int):
        path = self.df[self.datatype + "_path"].iloc[index]
        with TiffFile(path) as img:
            return zarr.open(img.aszarr())


class PatchDataset(Dataset):
    def __init__(self, dataset: Dataset, df_path: str, patch_size: int):
        """Dataset class for patch data (cropping).

        Args:
            dataset (Dataset): the train split dataset after torch.utils.data.randomsplit for valid and train
        """
        self.dataset = dataset
        self.df = read_parquet(df_path)
        self.patch_size = patch_size

    @cached_property
    def image(self) -> ZarrData:
        return ZarrData(self.df, "image")

    @cached_property
    def label(self) -> ZarrData:
        return ZarrData(self.df, "label")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """retrieves a random patch of the data with size patch_size

        Args:
            idx (int): index of the data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: patch of the data with size patch_size.

            Note: One may realize that for RFV, we may crop out some peripheral information that's correlated
            with the GT, but we neglect this correlation as the axial shearing from the off-axis microlenses
            is not significant.
        """

        # Recipe for fast dataloading with zarr courtesy of Mitchell Gilmore mgilm0re@bu.edu
        image = self.image[idx]
        label = self.label[idx]

        # uniformly sample a patch
        row_start = torch.randint(0, image.shape[-2] - self.patch_size, (1,))
        col_start = torch.randint(0, image.shape[-1] - self.patch_size, (1,))

        row_slice = slice(row_start, row_start + self.patch_size)
        col_slice = slice(col_start, col_start + self.patch_size)

        image = torch.from_numpy(
            image[row_slice, col_slice].astype(np.float32) / 0xFFFF
        )
        label = torch.from_numpy(label[row_slice, col_slice].astype(np.float32) / 0xFF)

        return image[None, ...], label[None, ...]

    def __len__(self):
        return len(self.dataset)


now = datetime.datetime.now()

timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")


class Trainer:
    def __init__(
        self,
        model: Module,
        config: dict,
    ):
        self.config = config
        self.model = model
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.model_dir = config["model_dir"]
        self.lowest_val_loss = float("inf")
        self.training_losses = []
        self.validation_losses = []
        self.random_seed = config.get("random_seed", None)
        self.use_amp = config.get("use_amp", False)
        self.optimizer_name = config.get("optimizer", "adam")
        self.lr_scheduler_name = config.get("lr_scheduler", "cosine_annealing")
        self.criterion_name = config.get("loss_criterion", "bce_with_logits")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
        print("DEVICE: ", self.device)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.logger.debug(f"Using device: {self.device}")
        self.scaler = (
            GradScaler() if self.use_amp else None
        )  # Initialize the scaler if using AMP

        # Initialize the loss criterion based on the configuration
        if self.criterion_name == "bce_with_logits":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.criterion_name == "mse":
            self.criterion = nn.MSELoss()
        elif self.criterion_name == "mae":
            self.criterion = nn.L1Loss()
        else:
            print(
                f"Unknown loss criterion: {self.criterion_name}. Using BCEWithLogitsLoss."
            )
        self.train_data_loader, self.val_data_loader = self._get_dataloaders()

    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        def split_dataset(dataset, split_ratio):
            dataset_size = len(dataset)
            train_size = int(split_ratio * dataset_size)
            val_size = dataset_size - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            return train_dataset, val_dataset

        complete_dataset: Dataset = CustomDataset(self.config["dataset_pq"])
        split_ratio = self.config["train_split"]
        train_dataset, val_dataset = split_dataset(complete_dataset, split_ratio)

        # only train_dataset is a PatchDataset. val_dataset is full sized images.
        train_dataset = PatchDataset(
            dataset=train_dataset,
            df_path=self.config["dataset_pq"],
            patch_size=self.config["patch_size"],
        )
        val_dataset = PatchDataset(
            dataset=val_dataset,
            df_path=self.config["dataset_pq"],
            patch_size=self.config["patch_size"],
        )

        train_dataloader = DataLoader(
            train_dataset, self.config.get("batch_size"), shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, self.config["batch_size"], shuffle=True
        )

        return train_dataloader, val_dataloader

    def _set_random_seed(self):
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)

    def _initialize_optimizer(self):
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=0.9
            )
        else:
            print(f"Unknown optimizer: {self.optimizer_name}. Using Adam.")
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _initialize_lr_scheduler(self, optimizer):
        if self.lr_scheduler_name == "cosine_annealing":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config["cosine_annealing_T_max"]
            )
        elif self.lr_scheduler_name == "step_lr":
            # StepLR scheduler
            step_size = self.config.get("step_lr_step_size", 10)
            gamma = self.config.get("step_lr_gamma", 0.5)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif self.lr_scheduler_name == "plateau":
            # Plateau scheduler
            patience = self.config.get("plateau_patience", 10)
            factor = self.config.get("plateau_factor", 0.5)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=patience, factor=factor, verbose=True
            )
        else:
            print(
                f"Unknown LR scheduler: {self.lr_scheduler_name}. Using Cosine Annealing."
            )
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return scheduler

    def train(self):
        model_name = f"sbrnet_{timestamp}.pt"
        self.model.to(self.device)
        self._set_random_seed()

        optimizer = self._initialize_optimizer()
        scheduler = self._initialize_lr_scheduler(optimizer)
        start_time = time.time()

        if self.use_amp:
            print("Using mixed-precision training with AMP.")

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for image, label in self.train_data_loader:
                # print(image.shape, label.shape)
                image, label = (
                    image.to(self.device),
                    label.to(self.device),
                )

                optimizer.zero_grad()

                if self.use_amp:
                    with autocast():
                        output = self.model(image)
                        loss = self.criterion(output, label)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    output = self.model(image)
                    loss = self.criterion(output, label)
                    loss.backward()
                    optimizer.step()
                self.logger.debug(
                    f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item()}"
                )

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_data_loader)
            self.training_losses.append(avg_train_loss)
            self.logger.debug(
                f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {avg_train_loss}"
            )

            val_loss = self.validate()
            self.validation_losses.append(val_loss)
            self.logger.debug(
                f"Epoch [{epoch + 1}/{self.epochs}], Validation Loss: {val_loss}"
            )

            if self.lr_scheduler_name == "plateau":
                scheduler.step(
                    val_loss
                )  # For Plateau scheduler, pass validation loss as an argument
            else:
                scheduler.step()

            if val_loss < self.lowest_val_loss:
                self.lowest_val_loss = val_loss
                save_state = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_losses": self.training_losses,
                    "validation_losses": self.validation_losses,
                    "time_elapsed": time.time() - start_time,
                }

                save_state.update(self.config)
                model_save_path = os.path.join(self.model_dir, model_name)
                torch.save(save_state, model_save_path)
                self.logger.debug("Model saved at epoch {}".format(epoch + 1))

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for image, label in self.val_data_loader:
                image, label = (
                    image.to(self.device),
                    label.to(self.device),
                )
                output = self.model(image)
                loss = self.criterion(output, label)
                total_loss += loss.item()
        return total_loss / len(self.val_data_loader)


# One-time script to generate dataframe parquet to map index to files


BASE_DIR = os.path.join(
    "/ad/eng/research/eng_research_cisl/jalido/kaggle/vessel_segmentatio/"
)
train_dir_path = os.path.join(BASE_DIR, "train")

folders = os.listdir(train_dir_path)

subfolders = ["images", "labels"]

df = pd.DataFrame()


# for folder in folders:
#     if folder == "kidney_3_dense":
#         # This folder only contains labels
#         continue

#     for subfolder in subfolders:
#         subfolder_path = os.path.join(train_dir_path, folder, subfolder)
#         files = os.listdir(subfolder_path)
#         for file in files:
#             file_path = os.path.join(subfolder_path, file)
#             if subfolder == "images":
#                 _df = pd.DataFrame.from_dict({"image_path": [file_path], "label_path": [None]})
#                 df = pd.concat([_df, df], ignore_index=True)
#             else:
#                 image_file_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), "images", file)
#                 df.loc[df["image_path"] == image_file_path, "label_path"] = file_path

# df.to_parquet(os.path.join(train_dir_path, "train_df.parquet"))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["NVIDIA_VISIBLE_DEVICES"] = "0"

now = datetime.datetime.now()

timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = f"/projectnb/tianlabdl/jalido/segmentation/sennet-hoa/.log/logging/hoa_net_{timestamp}.log"
logging.basicConfig(filename=log_file_path, level=logging.DEBUG)

args_dict = {
    "dataset_pq": os.path.join(train_dir_path, "train_df.parquet"),
    "model_dir": os.path.join(BASE_DIR, "model_dir", f"{timestamp}.pt"),
    "train_split": 0.8,
    "batch_size": 128,
    "learning_rate": 0.0001,
    "epochs": 20000,
    "weight_init": "kaiming_normal",
    "random_seed": 42,
    "optimizer": "sgd",
    "criterion_name": "bce_with_logits",
    "use_amp": False,
    "lr_scheduler": "cosine_annealing",
    "cosine_annealing_T_max": 30,
    "patch_size": 224,
}

model = torch.hub.load(
    "mateuszbuda/brain-segmentation-pytorch",
    "unet",
    in_channels=1,
    out_channels=1,
    init_features=32,
    pretrained=False,
)

trainer = Trainer(model, args_dict)

logging.info("Starting training...")

trainer.train()

logging.info("Training complete.")

logging.shutdown()

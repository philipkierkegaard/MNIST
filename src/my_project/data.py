from pathlib import Path

import torch
import typer
from torch.utils.data import Dataset


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()

def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    # exchange with the corrupted mnist dataset

    DATA_PATH = 'data/raw'
    train_images, train_targets = [], []

    for i in range(6):
        train_images.append(torch.load(f'{DATA_PATH}/train_images_{i}.pt', weights_only=True))
        train_targets.append(torch.load(f'{DATA_PATH}/train_target_{i}.pt', weights_only=True))

    train_images = torch.cat(train_images)
    train_targets = torch.cat(train_targets)

    test_images: torch.Tensor = torch.load(f'{DATA_PATH}/test_images.pt', weights_only=True)
    test_targets: torch.Tensor = torch.load(f'{DATA_PATH}/test_target.pt', weights_only=True)
    
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_targets = train_targets.long()
    test_targets = test_targets.long()

    train_set = torch.utils.data.TensorDataset(train_images, train_targets)
    test_set = torch.utils.data.TensorDataset(test_images, test_targets)


    return train_set, test_set


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        train_images, train_target = [], []
        for i in range(6):
            train_images.append(torch.load(f"{self.data_path}/train_images_{i}.pt"))
            train_target.append(torch.load(f"{self.data_path}/train_target_{i}.pt"))
        train_images = torch.cat(train_images)
        train_target = torch.cat(train_target)

        test_images: torch.Tensor = torch.load(f"{self.data_path}/test_images.pt")
        test_target: torch.Tensor = torch.load(f"{self.data_path}/test_target.pt")

        train_images = train_images.unsqueeze(1).float()
        test_images = test_images.unsqueeze(1).float()
        train_target = train_target.long()
        test_target = test_target.long()

        train_images = normalize(train_images)
        test_images = normalize(test_images)

        torch.save(train_images, f"{output_folder}/train_images.pt")
        torch.save(train_target, f"{output_folder}/train_target.pt")
        torch.save(test_images, f"{output_folder}/test_images.pt")
        torch.save(test_target, f"{output_folder}/test_target.pt")


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)

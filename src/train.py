import random
import shutil
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn import CNN

from Augmentation import augmentation


def parse_args():
    parser = ArgumentParser(
        prog="Train",
        description="Training program.",
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the input dataset.",
    )

    return parser.parse_args()


def data(src_path, dst_path, train_percent=0.80):
    if not (0.0 < train_percent < 1.0):
        raise ValueError("train_percent must be between 0.0 and 1.0")
    val_percent = 1.0 - train_percent
    dst_path = Path(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)
    train_dir = dst_path / "train_temp"
    val_dir = dst_path / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(src_path)
    for entry in src_path.iterdir():
        if entry.is_dir():
            src_dir = entry
            class_name = entry.name
            train_class_dir = train_dir / class_name
            val_class_dir = val_dir / class_name
            train_class_dir.mkdir(parents=True, exist_ok=True)
            val_class_dir.mkdir(parents=True, exist_ok=True)
            img_files = [
                f.name
                for f in src_dir.iterdir()
                if f.name.lower().endswith(".jpg")
            ]
            random.shuffle(img_files)
            n_train = max(1, int(len(img_files) * train_percent))
            n_val = max(1, int(len(img_files) * val_percent))
            train_files = img_files[:n_train]
            val_files = img_files[n_train : n_train + n_val]

            for fname in train_files:
                src_file = src_dir / fname
                shutil.copy2(src_file, train_class_dir)

            for fname in val_files:
                src_file = src_dir / fname
                shutil.copy2(src_file, val_class_dir)

    augmentation(train_dir, dst_path / "train")
    shutil.rmtree(train_dir)


def train(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], \
              Loss: {epoch_loss:.4f}, \
              Accuracy: {accuracy:.2f}%"
        )

    print("Training completed !")


def main():
    args = parse_args()

    try:
        if not Path(args.input).is_dir():
            raise FileNotFoundError

        with tempfile.TemporaryDirectory() as tmp_dir:
            data(args.input, tmp_dir)

            image_transform = transforms.Compose(
                [
                    transforms.Resize((150, 150)),
                    transforms.ToTensor(),
                ]
            )

            train_data = datasets.ImageFolder(
                root=str(Path(tmp_dir) / "train"),
                transform=image_transform,
            )

            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model = CNN(num_classes=len(train_data.classes)).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            train(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                num_epochs=10,
            )

            model_path = Path(tmp_dir) / "model.pth"
            torch.save(model.state_dict(), model_path)

            zip_path = shutil.make_archive("leaffliction", "zip", tmp_dir)
            print(f"Saved zip: {zip_path}")

    except FileNotFoundError:
        print(f"Error: Folder '{args.input}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for '{args.input}'.")
    except KeyboardInterrupt:
        print("Leaffliction: CTRL+C sent by user.")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()

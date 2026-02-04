import os
import random
import shutil
import signal
import tempfile
from argparse import ArgumentParser
from types import SimpleNamespace as Namespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn import CNN
from Transformation import transform


def parse_args():
    parser = ArgumentParser(
        prog="Train",
        description="Training program.",
    )

    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="Path to the input folder.",
    )

    return parser.parse_args()


def data(src_path, dst_path, train_percent=0.7):
    if not (0.0 < train_percent < 1.0):
        raise ValueError("train_percent must be between 0.0 and 1.0")
    val_percent = 1.0 - train_percent
    os.makedirs(dst_path, exist_ok=True)
    train_dir = os.path.join(dst_path, "train")
    val_dir = os.path.join(dst_path, "validation")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    for entry in os.scandir(src_path):
        if entry.is_dir():
            src_dir = entry.path
            class_name = entry.name
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)
            t_args = Namespace(
                mask=True,
                blur=True,
                histogram=False,
                roi=True,
                edge=True,
                pseudolandmarks=False,
            )
            img_files = [
                fname
                for fname in os.listdir(src_dir)
                if fname.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
                )
            ]
            random.shuffle(img_files)
            n_train = max(1, int(len(img_files) * train_percent))
            n_val = max(1, int(len(img_files) * val_percent))
            train_files = img_files[:n_train]
            val_files = img_files[n_train : n_train + n_val]

            for fname in train_files:
                src_file = os.path.join(src_dir, fname)
                shutil.copy2(src_file, train_class_dir)
                transform(
                    src_file,
                    train_class_dir,
                    True,
                    t_args,
                )

            for fname in val_files:
                src_file = os.path.join(src_dir, fname)
                shutil.copy2(src_file, val_class_dir)


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
        signal.signal(
            signal.SIGINT,
            lambda *_: (
                print("\033[2DLeaffliction: CTRL+C sent by user."),
                exit(1),
            ),
        )

        if not os.path.isdir(args.path):
            raise FileNotFoundError

        with tempfile.TemporaryDirectory() as tmp_dir:
            data(args.path, tmp_dir)

            image_transform = transforms.Compose(
                [
                    transforms.Resize((150, 150)),
                    transforms.ToTensor(),
                ]
            )

            train_data = datasets.ImageFolder(
                root=os.path.join(tmp_dir, "train"),
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

            model_path = os.path.join(tmp_dir, "model.pth")
            torch.save(model.state_dict(), model_path)

            zip_path = shutil.make_archive("leaffliction", "zip", tmp_dir)
            print(f"Saved zip: {zip_path}")

    except FileNotFoundError:
        print(f"Error: Folder '{args.path}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for '{args.path}'.")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()

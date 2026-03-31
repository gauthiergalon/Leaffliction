import random
import shutil
from argparse import ArgumentParser
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt

DEFAULT_INPUT_LOCATION = Path("images/")
DEFAULT_OUTPUT_LOCATION = Path("augmented_directory/")
NUMBER_TRANSFORMATION = 6


def augment(image, transform):
    return transform(image=image)["image"] if transform else image


def save(output_path, image):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)


def copy(images_path, input_dir, output_dir):
    for class_name, files in images_path.items():
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

        for file in files:
            src = input_dir / class_name / file
            dst = output_dir / class_name / file
            shutil.copy2(src, dst)


def balance_classes(input, inplace=False):
    categories = {
        entry.name: [
            file.name
            for file in entry.iterdir()
            if file.is_file() and file.suffix.lower() == ".jpg"
        ]
        for entry in input.iterdir()
        if entry.is_dir()
    }

    to_copy = {key: [] for key in categories.keys()}
    to_augment = {key: [] for key in categories.keys()}

    if inplace:
        target = max(
            (len(category) for category in categories.values()), default=0
        )
        for key, val in categories.items():
            size = len(val)
            diff = target - size
            num_to_augment = int(diff // 6)
            if num_to_augment > 0:
                if num_to_augment <= size:
                    to_augment[key] = random.sample(val, num_to_augment)
                else:
                    to_augment[key] = random.choices(val, k=num_to_augment)
    else:
        target = int(
            sum(len(category) for category in categories.values())
            / (len(categories) or 1)
        )

        for key, val in categories.items():
            size = len(val)
            if size >= target:
                to_copy[key] = random.sample(val, target)
            elif size >= target - NUMBER_TRANSFORMATION + 1:
                to_copy[key] = val
            else:
                diff = target - size
                num_to_augment = max(1, int(diff // NUMBER_TRANSFORMATION))
                num_to_augment = min(num_to_augment, size)
                to_augment[key] = random.sample(val, num_to_augment)
                to_copy[key] = [x for x in val if x not in to_augment[key]]

    return to_copy, to_augment


def display_images(images, labels):
    _, axes = plt.subplots(1, NUMBER_TRANSFORMATION + 1, figsize=(20, 4))

    for ax, img, title in zip(axes.flat, images, labels):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = ArgumentParser(
        prog="Augmentation",
        description=(
            "Applies image augmentations. "
            "If the input is a single image, it displays and "
            "saves augmented images."
            "If the input is a directory with subdirectories, it balances "
            "across classes and saves them."
        ),
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_LOCATION,
        type=Path,
        help="Path to the input image or directory containing images.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_LOCATION,
        type=Path,
        help="Path to the directory to save augmented images.",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help=(
            "Evaluation mode: uses a new separate output directory"
            "and balances by subsampling large classes."
        ),
    )

    return parser.parse_args()


def augmentation(input, output, eval_mode=False):
    try:
        if not input.exists():
            raise FileNotFoundError

        if eval_mode:
            inplace = False
        else:
            output = input if input.is_dir() else input.parent
            inplace = True

        if input.is_dir():
            is_file = False
            images_copy, images_augment = balance_classes(input, inplace)
            if not inplace:
                copy(images_copy, input, output)
        else:
            is_file = True
            images_augment = {input.parent.name: [input.name]}

        augmentation_names = [
            "Original",
            "Blur",
            "Contrast",
            "Crop",
            "Flip",
            "Rotate",
            "Shear",
        ]

        augmentation_effects = [
            None,
            A.Blur(blur_limit=(3, 7), p=1),
            A.RandomBrightnessContrast(contrast_limit=(0.75, 1.0), p=1),
            None,
            A.HorizontalFlip(p=1),
            A.Rotate(limit=45, p=1),
            A.Affine(shear=15, p=1),
        ]

        images = []

        for class_name, file_names in images_augment.items():
            for file_name in file_names:
                base_name, extension = (
                    Path(file_name).stem,
                    Path(file_name).suffix,
                )

                image = cv2.imread(
                    input if is_file else input / class_name / file_name
                )

                augmentation_effects[3] = A.RandomCrop(
                    height=int(image.shape[0] * 0.8),
                    width=int(image.shape[1] * 0.8),
                    p=1,
                )

                for augmentation_name, augmentation_effect in zip(
                    augmentation_names, augmentation_effects
                ):
                    augmented_image = augment(image, augmentation_effect)
                    if inplace and is_file:
                        out_path = (
                            output
                            / f"{base_name}_{augmentation_name}{extension}"
                        )
                    else:
                        out_path = (
                            output
                            / class_name
                            / f"{base_name}_{augmentation_name}{extension}"
                        )

                    save(out_path, augmented_image)
                    if is_file:
                        images.append(augmented_image)

                if is_file:
                    display_images(images, augmentation_names)

    except FileNotFoundError:
        print(f"Error: File or directory '{input}' not found.")
    except FileExistsError:
        print(f"Error: Directory '{output}' already exists.")
    except PermissionError:
        print(f"Error: Permission denied for '{input}'.")
    except KeyboardInterrupt:
        print("Leaffliction: CTRL+C sent by user.")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    args = parse_args()
    augmentation(args.input, args.output, args.eval)

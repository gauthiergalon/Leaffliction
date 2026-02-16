from argparse import ArgumentParser
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
import cv2
import shutil
import random

DEFAULT_INPUT_LOCATION = Path("images/")
DEFAULT_OUTPUT_LOCATION = Path("augmented_directory/")


def parse_args():
    parser = ArgumentParser(
        prog="Augmentation",
        description=(
            "Applies image augmentations. "
            "If the input is a single image, it displays and saves augmented images. "
            "If the input is a directory with subdirectories, it balances images across classes and saves them."
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

    return parser.parse_args()


def augment(image, transform):
    if transform:
        return transform(image=image)["image"]
    else:
        return image


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


def balance_classes(input):
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

    target = int(
        sum(len(category) for category in categories.values())
        / len(categories)
    )

    for key, val in categories.items():
        size = len(val)
        if size > target - 7:
            to_copy[key] = random.sample(val, target)
        elif size < target - 7:
            diff = target - size
            to_augment[key] = random.sample(val, int(diff // 6))
            to_copy[key] = [x for x in val if x not in to_augment[key]]
        else:
            to_copy[key] = val

    return to_copy, to_augment


def display_images(images, labels):
    _, axes = plt.subplots(1, 7, figsize=(20, 4))

    for ax, img, title in zip(axes.flat, images, labels):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def unique_directory(directory_path):
    if not directory_path.exists():
        return directory_path

    counter = 1
    while True:
        candidate = directory_path.with_name(
            f"{directory_path.name}_{counter}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def main():
    args = parse_args()

    try:
        if not args.input.exists():
            raise FileNotFoundError

        args.output = unique_directory(args.output)

        if args.input.is_dir():
            is_file = False
            images_copy, images_augment = balance_classes(args.input)
            copy(images_copy, args.input, args.output)
        else:
            is_file = True
            images_augment = {args.input.parent.name: [args.input.name]}

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
                    args.input
                    if is_file
                    else args.input / class_name / file_name
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
                    save(
                        args.output
                        / class_name
                        / f"{base_name}_{augmentation_name}{extension}",
                        augmented_image,
                    )
                    if is_file:
                        images.append(augmented_image)

                if is_file:
                    display_images(images, augmentation_names)

    except FileNotFoundError:
        print(f"Error: File or directory '{args.input}' not found.")
    except FileExistsError:
        print(f"Error: Directory '{args.output}' already exists.")
    except PermissionError:
        print(f"Error: Permission denied for '{args.input}'.")
    except KeyboardInterrupt:
        print("Leaffliction: CTRL+C sent by user.")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()

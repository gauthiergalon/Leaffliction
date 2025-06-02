import os
from argparse import ArgumentParser

import albumentations as A
import cv2


def parse_args():
    parser = ArgumentParser(
        prog="Augmentation",
        description="Image augmentation.",
    )

    parser.add_argument(
        "--path",
        type=str,
        help="Path to the input image.",
    )

    return parser.parse_args()


def apply_and_save(image, transform, output_path):
    augmented = transform(image=image)
    transformed_image = augmented["image"]
    cv2.imwrite(output_path, transformed_image)


def main():
    args = parse_args()

    try:
        folder = os.path.dirname(args.path)
        base_name = os.path.splitext(os.path.basename(args.path))[0]
        ext = os.path.splitext(os.path.basename(args.path))[1]

        image = cv2.imread(args.path)

        blur = A.Blur(blur_limit=(3, 7), p=1)
        apply_and_save(
            image,
            blur,
            os.path.join(folder, f"{base_name}_Blur{ext}"),
        )

        high_contrast = A.RandomBrightnessContrast(
            contrast_limit=(0.75, 1.0), p=1
        )
        apply_and_save(
            image,
            high_contrast,
            os.path.join(folder, f"{base_name}_Contrast{ext}"),
        )

        crop = A.RandomCrop(
            height=int(image.shape[0] * 0.8),
            width=int(image.shape[1] * 0.8),
            p=1,
        )
        apply_and_save(
            image, crop, os.path.join(folder, f"{base_name}_Crop{ext}")
        )

        flip = A.HorizontalFlip(p=1)
        apply_and_save(
            image, flip, os.path.join(folder, f"{base_name}_Flip{ext}")
        )

        rotate = A.Rotate(limit=45, p=1)
        apply_and_save(
            image, rotate, os.path.join(folder, f"{base_name}_Rotate{ext}")
        )

        shear = A.Affine(shear=15, p=1)
        apply_and_save(
            image, shear, os.path.join(folder, f"{base_name}_Shear{ext}")
        )

    except FileNotFoundError:
        print(f"Error: File '{args.path}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for '{args.path}'.")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()

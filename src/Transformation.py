import os
import signal
from argparse import ArgumentParser

import cv2
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv


def parse_args():
    parser = ArgumentParser(
        prog="Transformation",
        description="Image transformation.",
    )

    parser.add_argument(
        "--src",
        type=str,
        help="Path to the input image or images folder.",
    )

    parser.add_argument(
        "--dst",
        type=str,
        required=False,
        help="Path to the output image(s).",
    )

    return parser.parse_args()


def img_color_histogram(rgb_img, mask, save=False):
    total_masked_pixels = cv2.countNonZero(mask)
    if total_masked_pixels == 0:
        print("Warning: Mask is empty, cannot generate percentage histogram.")
        return

    _, rgb_data = pcv.visualize.histogram(
        img=rgb_img,
        mask=mask,
        hist_data=True,
    )
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    _, hsv_data = pcv.visualize.histogram(
        img=hsv_img,
        mask=mask,
        hist_data=True,
    )
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)
    _, lab_data = pcv.visualize.histogram(
        img=lab_img,
        mask=mask,
        hist_data=True,
    )

    all_data = [
        (rgb_data, ["blue", "green", "red"], "RGB"),
        (hsv_data, ["hue", "saturation", "value"], "HSV"),
        (lab_data, ["L*", "a*", "b*"], "LAB"),
    ]

    plt.figure(figsize=(15, 7))

    colors = [
        "blue",
        "green",
        "red",
        "purple",
        "cyan",
        "orange",
        "grey",
        "pink",
        "yellow",
    ]

    color_index = 0
    for data_tuple in all_data:
        hist_data, channel_names, space_name = data_tuple
        channels_in_space = hist_data["color channel"].unique()

        for i, channel_key in enumerate(channels_in_space):
            channel_plot_data = hist_data[
                hist_data["color channel"] == channel_key
            ].copy()

            channel_plot_data["hist_percentage"] = (
                channel_plot_data["hist_count"] / total_masked_pixels
            ) * 100

            plt.plot(
                channel_plot_data["pixel intensity"],
                channel_plot_data["hist_percentage"],
                label=f"{channel_names[i]} ({space_name})",
                color=colors[color_index],
            )
            color_index += 1

    plt.xlabel("Pixel intensity")
    plt.ylabel("Proportion of pixels (%)")
    plt.legend(loc="center right")
    plt.grid(True)
    plt.show()


def img_analyze_object(rgb_img, mask, save=False):
    rect_roi = pcv.roi.rectangle(img=rgb_img, x=0, y=0, h=256, w=256)
    cleaned_mask = pcv.fill(bin_img=mask, size=50)
    filtered_mask = pcv.roi.filter(
        mask=cleaned_mask, roi=rect_roi, roi_type="partial"
    )
    shape_img = pcv.analyze.size(img=rgb_img, labeled_mask=filtered_mask)

    pcv.plot_image(img=shape_img)


def img_gaussian_blur(rgb_img, mask, save=False):
    gaussian_img = pcv.gaussian_blur(
        img=mask,
        ksize=(3, 3),
    )
    pcv.plot_image(img=gaussian_img)


def img_apply_mask(rgb_img, mask, save=False):
    masked_image = pcv.apply_mask(img=rgb_img, mask=mask, mask_color="white")
    pcv.plot_image(img=masked_image)


def img_pseudolandmarks(rgb_img, mask, save=False):
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=rgb_img,
        mask=mask,
    )


def directory(args):
    if not os.path.isdir(args.dst):
        raise SystemError("Error: missing argument.")


def file(args):
    rgb_img, _, _ = pcv.readimage(args.src)

    a_gray = pcv.rgb2gray_lab(rgb_img=rgb_img, channel="a")
    a_mask = pcv.threshold.otsu(gray_img=a_gray, object_type="dark")

    v_gray = pcv.rgb2gray_hsv(rgb_img=rgb_img, channel="v")
    v_mask = pcv.threshold.otsu(gray_img=v_gray, object_type="dark")

    # img_apply_mask(rgb_img, a_mask)
    # img_gaussian_blur(rgb_img, v_mask)
    img_color_histogram(rgb_img, a_mask)
    # img_analyze_object(rgb_img, a_mask)
    # img_pseudolandmarks(rgb_img, a_mask)


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

        if os.path.isdir(args.src):
            directory(args)
        elif os.path.isfile(args.src):
            file(args)
        else:
            raise FileNotFoundError

    except FileNotFoundError:
        print(f"Error: File '{args.src}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for '{args.src}'.")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()

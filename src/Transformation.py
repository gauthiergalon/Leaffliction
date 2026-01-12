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

    parser.add_argument(
        "--mask",
        action="store_true",
        help="Generate mask effect.",
    )

    parser.add_argument(
        "--blur",
        action="store_true",
        help="Generate Gaussian blur effect.",
    )

    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Generate color histograms.",
    )

    parser.add_argument(
        "--roi",
        action="store_true",
        help="Generate region of interest (ROI).",
    )

    parser.add_argument(
        "--edge",
        action="store_true",
        help="Generate Detect edges.",
    )

    parser.add_argument(
        "--pseudolandmarks",
        action="store_true",
        help="Generate pseudolandmarks.",
    )

    return parser.parse_args()


def img_color_histogram(rgb_img, mask):
    total_masked_pixels = cv2.countNonZero(mask)
    if total_masked_pixels == 0:
        return None

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

    fig = plt.figure(figsize=(15, 7))

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

            plt.plot(
                channel_plot_data["pixel intensity"],
                channel_plot_data["proportion of pixels (%)"],
                label=f"{channel_names[i]} ({space_name})",
                color=colors[color_index],
            )
            color_index += 1

    plt.xlabel("Pixel intensity")
    plt.ylabel("Proportion of pixels (%)")
    plt.legend(loc="center right")
    plt.grid(True)

    return fig


def img_roi(rgb_img, mask):
    height, width, _ = rgb_img.shape
    rect_roi = pcv.roi.rectangle(img=rgb_img, x=0, y=0, h=height, w=width)
    cleaned_mask = pcv.fill(bin_img=mask, size=50)
    filtered_mask = pcv.roi.filter(
        mask=cleaned_mask, roi=rect_roi, roi_type="partial"
    )
    shape_img = pcv.analyze.size(img=rgb_img, labeled_mask=filtered_mask)
    return shape_img


def img_analyze_thermal(rgb_img, mask):
    gray_img = pcv.rgb2gray(rgb_img=rgb_img)
    pseudocolor_img = pcv.visualize.pseudocolor(
        gray_img,
        min_value=0,
        max_value=100,
        mask=mask,
    )
    return pseudocolor_img


def img_canny_edge(rgb_img):
    edges_img = pcv.canny_edge_detect(rgb_img, sigma=2)
    return edges_img


def img_gaussian_blur(mask):
    gaussian_img = pcv.gaussian_blur(
        img=mask,
        ksize=(3, 3),
    )
    return gaussian_img


def img_apply_mask(rgb_img, mask):
    mask_image = pcv.apply_mask(img=rgb_img, mask=mask, mask_color="white")
    return mask_image


def img_pseudolandmarks(rgb_img, mask):
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=rgb_img,
        mask=mask,
    )
    return top, bottom, center_v


def extract_coordinates(landmarks):
    x_coords = [landmark[0][0] for landmark in landmarks]
    y_coords = [landmark[0][1] for landmark in landmarks]

    return x_coords, y_coords


def generate_pseudolandmarks(rgb_img, top, bottom, center_v):
    top_x, top_y = extract_coordinates(top)
    bottom_x, bottom_y = extract_coordinates(bottom)
    center_v_x, center_v_y = extract_coordinates(center_v)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    plt.scatter(
        top_x,
        top_y,
        color="blue",
        label="Top Landmark",
        s=100,
    )
    plt.scatter(
        bottom_x,
        bottom_y,
        color="magenta",
        label="Bottom Landmark",
        s=100,
    )
    plt.scatter(
        center_v_x,
        center_v_y,
        color="orange",
        label="Center Vertical Landmark",
        s=100,
    )
    plt.legend(loc="upper right")
    plt.title("Pseudolandmarks on Image")
    plt.axis("off")
    return fig


def transform(src, dst, save, args):
    rgb_img, _, _ = pcv.readimage(src)

    a_gray = pcv.rgb2gray_lab(rgb_img=rgb_img, channel="a")
    a_mask = pcv.threshold.otsu(gray_img=a_gray, object_type="dark")

    v_gray = pcv.rgb2gray_hsv(rgb_img=rgb_img, channel="v")
    v_mask = pcv.threshold.otsu(gray_img=v_gray, object_type="dark")

    if not any(
        [
            args.mask,
            args.blur,
            args.histogram,
            args.roi,
            args.edge,
            args.pseudolandmarks,
        ]
    ):
        args.mask = True
        args.blur = True
        args.histogram = True
        args.roi = True
        args.edge = True
        args.pseudolandmarks = True

    abs_path = None
    if save:
        abs_path = os.path.abspath(dst) if dst else os.getcwd()
        os.makedirs(abs_path, exist_ok=True)
        pcv.params.debug_outdir = abs_path

    base_name = os.path.basename(src)

    if args.mask:
        mask_image = img_apply_mask(rgb_img, a_mask)
        if save:
            pcv.print_image(mask_image, f"{abs_path}/mask_{base_name}")
        else:
            pcv.plot_image(mask_image)

    if args.blur:
        gaussian_img = img_gaussian_blur(v_mask)
        if save:
            pcv.print_image(gaussian_img, f"{abs_path}/gaussian_{base_name}")
        else:
            pcv.plot_image(gaussian_img)

    if args.histogram:
        color_fig = img_color_histogram(rgb_img, a_mask)
        if save:
            color_fig.savefig(f"{abs_path}/color_{base_name}")
        else:
            plt.show(block=False)

    if args.roi:
        shape_img = img_roi(rgb_img, a_mask)
        if save:
            pcv.print_image(shape_img, f"{abs_path}/roi_{base_name}")
        else:
            pcv.plot_image(shape_img)

    if args.edge:
        edges_img = img_canny_edge(rgb_img)
        if save:
            pcv.print_image(edges_img, f"{abs_path}/edge_{base_name}")
        else:
            pcv.plot_image(edges_img)

    if args.pseudolandmarks:
        top, bottom, center_v = img_pseudolandmarks(rgb_img, a_mask)
        pseudo_fig = generate_pseudolandmarks(rgb_img, top, bottom, center_v)
        if save:
            pseudo_fig.savefig(f"{abs_path}/pseudo_{base_name}")
        else:
            plt.show(block=False)


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

        files = []

        if os.path.isdir(args.src):
            files = [
                os.path.join(args.src, f)
                for f in os.listdir(args.src)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
                )
            ]
        elif os.path.isfile(args.src):
            if args.src.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
            ):
                files = [args.src]
            else:
                raise ValueError(f"'{args.src}' is not a valid image file.")

        else:
            raise FileNotFoundError

        if args.dst:
            os.makedirs(os.path.abspath(args.dst), exist_ok=True)
            pcv.params.debug_outdir = os.path.abspath(args.dst)
            save = True
        else:
            save = False

        for file in files:
            transform(file, args.dst, save, args)

    except FileNotFoundError:
        print(f"Error: File '{args.src}' not found.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except PermissionError:
        print(f"Error: Permission denied for '{args.src}'.")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()

from argparse import ArgumentParser
from pathlib import Path

import cv2
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv

HIST_COLORS = [
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

HIST_CHANNELS = [
    (cv2.COLOR_BGR2RGB, ["blue", "green", "red"], "RGB"),
    (cv2.COLOR_BGR2HSV, ["hue", "saturation", "value"], "HSV"),
    (cv2.COLOR_BGR2LAB, ["L*", "a*", "b*"], "LAB"),
]


def parse_args():
    parser = ArgumentParser(
        prog="Transformation",
        description="Image transformation.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input image or images folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        help="Path to the output image(s).",
    )
    return parser.parse_args()


def get_histogram_data(rgb_img, mask):
    if cv2.countNonZero(mask) == 0:
        return None

    all_data = []
    for conversion, names, space in HIST_CHANNELS:
        converted = (
            rgb_img if space == "RGB" else cv2.cvtColor(rgb_img, conversion)
        )
        _, data = pcv.visualize.histogram(
            img=converted,
            mask=mask,
            hist_data=True,
        )
        all_data.append((data, names, space))
    return all_data


def plot_histogram(ax, all_data):
    color_index = 0
    for hist_data, channel_names, space_name in all_data:
        for i, ch_key in enumerate(hist_data["color channel"].unique()):
            ch_data = hist_data[hist_data["color channel"] == ch_key]
            ax.plot(
                ch_data["pixel intensity"],
                ch_data["proportion of pixels (%)"],
                label=f"{channel_names[i]} ({space_name})",
                color=HIST_COLORS[color_index],
            )
            color_index += 1
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(True)


def plot_pseudolandmarks(ax, rgb_img, top, bottom, center_v):
    ax.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    for landmarks, color, label in [
        (top, "blue", "Top"),
        (bottom, "magenta", "Bottom"),
        (center_v, "orange", "Center V"),
    ]:
        x = [lm[0][0] for lm in landmarks]
        y = [lm[0][1] for lm in landmarks]
        ax.scatter(x, y, color=color, label=label, s=50)
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")


def compute_masks(rgb_img):
    a_gray = pcv.rgb2gray_lab(rgb_img=rgb_img, channel="a")
    a_mask = pcv.threshold.otsu(gray_img=a_gray, object_type="dark")
    v_gray = pcv.rgb2gray_hsv(rgb_img=rgb_img, channel="v")
    v_mask = pcv.threshold.otsu(gray_img=v_gray, object_type="dark")
    return a_mask, v_mask


def compute_transformations(rgb_img, a_mask, v_mask):
    mask_image = pcv.apply_mask(img=rgb_img, mask=a_mask, mask_color="white")
    gaussian_img = pcv.gaussian_blur(img=v_mask, ksize=(3, 3))

    h, w = rgb_img.shape[:2]
    rect_roi = pcv.roi.rectangle(img=rgb_img, x=0, y=0, h=h, w=w)
    cleaned = pcv.fill(bin_img=a_mask, size=50)
    filtered = pcv.roi.filter(mask=cleaned, roi=rect_roi, roi_type="partial")
    shape_img = pcv.analyze.size(img=rgb_img, labeled_mask=filtered)

    edges_img = pcv.canny_edge_detect(rgb_img, sigma=2)

    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=rgb_img,
        mask=a_mask,
    )

    return (
        mask_image,
        gaussian_img,
        shape_img,
        edges_img,
        top,
        bottom,
        center_v,
    )


def save_images(
    output,
    base_name,
    rgb_img,
    a_mask,
    mask_image,
    gaussian_img,
    shape_img,
    edges_img,
    top,
    bottom,
    center_v,
):
    pcv.print_image(mask_image, f"{output}/mask_{base_name}")
    pcv.print_image(gaussian_img, f"{output}/gaussian_{base_name}")
    pcv.print_image(shape_img, f"{output}/roi_{base_name}")
    pcv.print_image(edges_img, f"{output}/edge_{base_name}")

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_pseudolandmarks(ax, rgb_img, top, bottom, center_v)
    ax.set_title("Pseudolandmarks on Image")
    fig.savefig(f"{output}/pseudo_{base_name}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    all_data = get_histogram_data(rgb_img, a_mask)
    if all_data:
        fig, ax = plt.subplots(figsize=(15, 7))
        plot_histogram(ax, all_data)
        fig.savefig(
            f"{output}/histogram_{base_name}",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def show_composite(
    rgb_img,
    a_mask,
    mask_image,
    gaussian_img,
    shape_img,
    edges_img,
    top,
    bottom,
    center_v,
):
    fig = plt.figure(figsize=(12, 12))

    panels = [
        (1, cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), "Original", {}),
        (2, cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB), "Mask", {}),
        (3, gaussian_img, "Gaussian Blur", {"cmap": "gray"}),
        (4, cv2.cvtColor(shape_img, cv2.COLOR_BGR2RGB), "ROI", {}),
        (5, edges_img, "Canny Edges", {"cmap": "gray"}),
    ]

    for pos, img, title, kwargs in panels:
        ax = fig.add_subplot(3, 3, pos)
        ax.imshow(img, **kwargs)
        ax.set_title(title)
        ax.axis("off")

    ax6 = fig.add_subplot(3, 3, 6)
    plot_pseudolandmarks(ax6, rgb_img, top, bottom, center_v)
    ax6.set_title("Pseudolandmarks")

    all_data = get_histogram_data(rgb_img, a_mask)
    if all_data:
        ax7 = fig.add_subplot(3, 3, (7, 9))
        plot_histogram(ax7, all_data)
        ax7.set_title("Color Histograms")

    plt.tight_layout()
    plt.show()


def transform(input, output, save):
    rgb_img, _, _ = pcv.readimage(input)
    a_mask, v_mask = compute_masks(rgb_img)
    results = compute_transformations(rgb_img, a_mask, v_mask)
    mask_image, gaussian_img, shape_img, edges_img, top, bottom, center_v = (
        results
    )

    if save:
        save_images(
            output,
            Path(input).name,
            rgb_img,
            a_mask,
            mask_image,
            gaussian_img,
            shape_img,
            edges_img,
            top,
            bottom,
            center_v,
        )
    else:
        show_composite(
            rgb_img,
            a_mask,
            mask_image,
            gaussian_img,
            shape_img,
            edges_img,
            top,
            bottom,
            center_v,
        )


def main():
    args = parse_args()

    try:
        if args.input.is_dir():
            files = [
                f for f in args.input.iterdir() if f.suffix.lower() == ".jpg"
            ]
            if not args.output:
                raise ValueError(
                    "Output path must be specified when input is a directory."
                )
            args.output.mkdir(parents=True, exist_ok=True)
            pcv.params.debug_outdir = args.output
            save = True
        elif args.input.is_file():
            if args.input.suffix.lower() != ".jpg":
                raise ValueError(f"'{args.input}' is not a valid image file.")
            files = [args.input]
            save = bool(args.output)
            if args.output:
                args.output.mkdir(parents=True, exist_ok=True)
                pcv.params.debug_outdir = args.output
        else:
            raise FileNotFoundError

        for file in files:
            transform(file, args.output, save)

    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except PermissionError:
        print(f"Error: Permission denied for '{args.input}'.")
    except KeyboardInterrupt:
        print("Leaffliction: CTRL+C sent by user.")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()

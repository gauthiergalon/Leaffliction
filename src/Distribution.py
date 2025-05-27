import os
import signal
from argparse import ArgumentParser

import matplotlib.pyplot as plt

DEFAULT_LOCATION_DATASET = "images/"


def parse_args():
    parser = ArgumentParser(
        prog="Distribution",
        description="Plot distribution for each plant.",
    )

    parser.add_argument(
        "--path",
        type=str,
        default=DEFAULT_LOCATION_DATASET,
        help=f"Path to the input dataset. \
                Defaults to '{DEFAULT_LOCATION_DATASET}' if not specified.",
    )

    return parser.parse_args()


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

        categories = {
            entry.name: len(
                [
                    file
                    for file in os.listdir(entry.path)
                    if os.path.isfile(os.path.join(entry.path, file))
                    and file.lower().endswith(".jpg")
                ]
            )
            for entry in os.scandir(args.path)
            if entry.is_dir()
        }

        cmap = plt.colormaps["tab10"]
        colors = [cmap(i) for i in range(len(categories))]

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Class Distribution")

        axs[0].pie(categories.values(), autopct="%1.0f%%", colors=colors)
        axs[1].bar(categories.keys(), categories.values(), color=colors)
        axs[1].set_xticks([])

        fig.legend(categories.keys(), loc="upper left")

        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{args.path}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for '{args.path}'.")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()

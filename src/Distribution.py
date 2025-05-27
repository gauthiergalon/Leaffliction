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
                ]
            )
            for entry in os.scandir(args.path)
            if entry.is_dir()
        }

        cmap = plt.cm.get_cmap("tab10", len(categories))
        colors = [cmap(i) for i in range(len(categories))]

        plt.figure(figsize=(16, 8))

        plt.subplot(121)
        plt.pie(
            categories.values(),
            labels=categories.keys(),
            autopct="%1.0f%%",
            colors=colors,
        )

        plt.subplot(122)
        plt.bar(categories.keys(), categories.values(), color=colors)

        plt.suptitle("Class Distribution")
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{args.path}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for '{args.path}'.")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()

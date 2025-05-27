import os
import signal
import matplotlib.pyplot as plt
from argparse import ArgumentParser

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
        help=f"Path to the input dataset. Defaults to '{DEFAULT_LOCATION_DATASET}' if not specified.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        signal.signal(
            signal.SIGINT,
            lambda *_: (print("\033[2DLeaffliction: CTRL+C sent by user."), exit(1)),
        )

        categories = {}

        for dirpath, _, filenames in os.walk(args.path):
            if os.path.abspath(dirpath) == os.path.abspath(args.path):
                continue
            categorie = os.path.basename(dirpath)
            categories[categorie] = len(filenames)

        cmap = plt.cm.get_cmap('tab10', len(categories))
        colors = [cmap(i) for i in range(len(categories))]

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Class Distribution")

        axs[0].pie(categories.values(), autopct='%1.0f%%', colors=colors)
        axs[1].bar(categories.keys(), categories.values(), color=colors)
        axs[1].set_xticks([])

        fig.legend(categories.keys(), loc="upper left")
        
        plt.show()

    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()

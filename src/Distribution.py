from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt

DEFAULT_INPUT_LOCATION = "images/"


def parse_args():
    parser = ArgumentParser(
        prog="Distribution",
        description="Plot distribution for each plant.",
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        default=DEFAULT_INPUT_LOCATION,
        help=f"Path to the input dataset. \
                Defaults to '{DEFAULT_INPUT_LOCATION}' if not specified.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        categories = {
            entry.name: len(
                [
                    file
                    for file in entry.iterdir()
                    if file.is_file() and file.suffix.lower() == ".jpg"
                ]
            )
            for entry in args.input.iterdir()
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
        print(f"Error: File '{args.input}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for '{args.input}'.")
    except KeyboardInterrupt:
        print("Leaffliction: CTRL+C sent by user.")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()

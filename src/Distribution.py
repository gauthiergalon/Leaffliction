# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Distribution.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/05/27 13:59:07 by ggalon            #+#    #+#              #
#    Updated: 2025/05/27 16:31:03 by ggalon           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

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

        plt.figure(figsize=(16, 8))

        plt.subplot(121)
        plt.pie(categories.values(), labels=categories.keys(), autopct='%1.0f%%', colors=colors)

        plt.subplot(122)
        plt.bar(categories.keys(), categories.values(), color=colors)

        plt.suptitle("Class Distribution")
        plt.show()

    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()

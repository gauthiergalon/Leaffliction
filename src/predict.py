from argparse import ArgumentParser
from pathlib import Path

import cv2
import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa E402
from PIL import Image  # noqa E402
from plantcv import plantcv as pcv  # noqa E402
from torchvision import transforms  # noqa E402

import utils  # noqa E402
from cnn import CNN  # noqa E402
from Transformation import compute_masks  # noqa E402


def parse_args():
    parser = ArgumentParser(
        prog="Predict",
        description="Prediction program for leaf disease classification.",
    )

    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to the trained model file (.pth).",
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the input image or directory containing images.",
    )

    return parser.parse_args()


def load_model_and_classes(model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        classes = checkpoint["classes"]
        state_dict = checkpoint["model_state_dict"]

        model = CNN(num_classes=len(classes))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, classes
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return None, None
    except KeyError as ex:
        print(f"Error: Invalid model format, missing key: {ex}")
        return None, None
    except Exception as ex:
        print(f"Error loading model: {ex}")
        return None, None


def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
            ]
        )
        return transform(image)
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        return None
    except Exception as ex:
        print(f"Error preprocessing image: {ex}")
        return None


def predict_single(model, image_tensor, classes, device):
    try:
        image_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = (
            classes[predicted.item()]
            if classes
            else f"Class {predicted.item()}"
        )
        confidence_score = confidence.item() * 100

        return predicted_class, confidence_score
    except Exception as ex:
        print(f"Error during prediction: {ex}")
        return None, None


def display_prediction(image_path, predicted_class, confidence):
    try:
        rgb_img, _, _ = pcv.readimage(str(image_path))

        a_mask, _ = compute_masks(rgb_img)

        mask_image = pcv.apply_mask(
            img=rgb_img, mask=a_mask, mask_color="white"
        )

        fig = plt.figure(figsize=(12, 8), facecolor="black")

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        ax1.axis("off")
        ax1.set_facecolor("black")

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB))
        ax2.axis("off")
        ax2.set_facecolor("black")

        ax3 = fig.add_subplot(2, 1, 2)
        ax3.set_facecolor("black")
        ax3.text(
            0.5,
            0.6,
            "=== DL classification ===",
            ha="center",
            va="center",
            fontsize=24,
            color="white",
            fontweight="bold",
            transform=ax3.transAxes,
        )
        ax3.text(
            0.5,
            0.3,
            f"Class predicted : {predicted_class}",
            ha="center",
            va="center",
            fontsize=20,
            color="lime",
            fontweight="bold",
            transform=ax3.transAxes,
        )
        ax3.axis("off")

        plt.tight_layout()
        utils.show_plot()

    except Exception as ex:
        print(f"Error displaying images: {ex}")


def predict(model_path, input_path):
    try:
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, classes = load_model_and_classes(model_path, device)
        if not model:
            return

        if input_path.is_file():
            image_tensor = preprocess_image(input_path)
            if image_tensor is not None:
                predicted_class, confidence = predict_single(
                    model, image_tensor, classes, device
                )
                if predicted_class:
                    print(f"Image: {input_path.name}")
                    print(f"Prediction: {predicted_class}")
                    print(f"Confidence: {confidence:.2f}%")
                    display_prediction(input_path, predicted_class, confidence)
        else:
            image_files = [
                f
                for f in input_path.rglob("*")
                if f.is_file()
                and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]

            if not image_files:
                print(f"No image files found in '{input_path}'.")
                return

            print(f"Found {len(image_files)} image(s) to process.\n")

            expected_class = None
            for image_path in image_files:
                parent_name = image_path.parent.name
                if parent_name in classes:
                    expected_class = parent_name
                    break

            total_predictions = 0
            correct_predictions = 0

            for image_path in image_files:
                image_tensor = preprocess_image(image_path)
                if image_tensor is not None:
                    predicted_class, confidence = predict_single(
                        model, image_tensor, classes, device
                    )
                    if predicted_class:
                        image_expected_class = image_path.parent.name
                        if image_expected_class not in classes:
                            image_expected_class = expected_class

                        is_correct = (
                            predicted_class == image_expected_class
                            if image_expected_class
                            else None
                        )

                        total_predictions += 1
                        if is_correct:
                            correct_predictions += 1

                        status = ""
                        if is_correct is not None:
                            status = "v" if is_correct else "x"

                        print(f"Image: {image_path.name}")
                        print(f"Expected: {image_expected_class}")
                        print(f"Prediction: {predicted_class} {status}")
                        print(f"Confidence: {confidence:.2f}%")
                        print()

            if total_predictions > 0 and expected_class:
                accuracy = (correct_predictions / total_predictions) * 100
                print("=" * 50)
                print("ACCURACY RESULTS:")
                print(f"Total images: {total_predictions}")
                print(f"Correct predictions: {correct_predictions}")
                print(
                    f"Incorrect predictions: \
                        {total_predictions - correct_predictions}"
                )
                print(f"Accuracy: {accuracy:.2f}%")
                print("=" * 50)

    except FileNotFoundError:
        print(f"Error: Path '{input_path}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for '{input_path}'.")
    except KeyboardInterrupt:
        print("Leaffliction: CTRL+C sent by user.")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


def main():
    args = parse_args()
    predict(args.model, args.input)


if __name__ == "__main__":
    main()

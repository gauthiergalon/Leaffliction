import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def show_plot():
    """
    Save the current matplotlib figure to a temporary file and open it.
    Uses Agg backend to avoid Qt conflicts with opencv.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    plt.savefig(tmp_path, bbox_inches="tight")
    plt.close()
    subprocess.Popen(["xdg-open", tmp_path])

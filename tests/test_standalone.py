from PIL import Image
from justpyplot import justpyplot as jplt
import numpy as np


def test_standalone():
    x = np.linspace(0, 4*np.pi, 100)
    y = np.sin(x) * np.exp(-x/10)

    figure, grid, labels, title = jplt.plot(
        np.array([x, y]),
        grid={'nticks': 20, 'color': (128, 128, 128, 255), 'label_color': (0, 0, 255, 255), 'label_font_size': 0.8},
        figure={'line_color': (255, 0, 0, 255), 'line_width': 2},
        title='Sine Wave',
        size=(300, 300),
        max_len=100
    )

    # Convert to uint8 for display
    blended = figure + grid + labels + title
    blended = np.array(blended)
    blended = blended.astype(np.uint8)
    blended = Image.fromarray(blended)
    blended.save("test.png")


if __name__ == "__main__":
    test_standalone()

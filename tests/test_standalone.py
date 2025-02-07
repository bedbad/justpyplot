from PIL import Image
from justpyplot import justpyplot as jplt
import numpy as np


def test_standalone():
    x = np.linspace(0, 4*np.pi, 100)
    y = np.sin(x) * np.exp(-x/10)

    figure, grid, labels, title = jplt.plot(
        np.array([x, y]),
        grid={'nticks': 10, 'color': (128, 128, 128, 255), 'label_color': (255, 0, 0, 255),'precision': 1, 'label_font_size': 0.9},
        figure={'scatter':False,'point_color': (255, 0, 0, 255), 'point_radius':3, 'line_color':(0,64,64, 255), 'line_width': 2, 'marker_style':'circle'},
        title='Sine Wave',
        size=(300, 400),
        max_len=100
    )

    # Convert to uint8 for display
    # blended = grid + figure + labels + title
    blended = jplt.blend(grid, figure, labels, title)
    blended = np.array(blended)
    blended = blended.astype(np.uint8)

    blended_rgb = np.dstack((blended[:, :, 2], blended[:, :, 1], blended[:, :, 0]))
    pil_image = Image.fromarray(blended_rgb)
    pil_image.save("test.png")


if __name__ == "__main__":
    test_standalone()

import numpy as np
import cv2
from justpyplot import justpyplot as jplt

print(jplt.__file__)

def test_plot():
    # Create a window
    cv2.namedWindow("Plot Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Plot Test", 600, 400)

    # Define plot options outside the loop
    grid_options = {
        'nticks': 10,
        'color': (128, 128, 128, 255),
        'label_color': (0, 0, 255, 255),
        'label_font_size': 0.8,
        'precision': 2,
        'title_margin': 30,
        'y_tick_offset': 5,
        'x_tick_offset': 5
    }

    figure_options = {
        'line_color': (255, 0, 0, 255),
        'line_width': 2,
        'scatter': False,
        'point_color': (0, 255, 0, 255),
        'point_radius': 3,
        'marker_style': 'circle',
        'line_thickness': 2,
        'line_style': 'solid'
    }

    for i in range(10000):
        # Generate sample data
        x = np.linspace(i * 0.1, 4 * np.pi+ i * 0.1, 100)
        y = np.sin(x) * np.exp(-x / 10)

        # Define the title for each frame
        title = f"Dynamic Plot (Frame {i})"

        # Call the plot function
        figure, grid, labels, title_img = jplt.plot(
            np.vstack((x, y)),
            grid=grid_options,
            figure=figure_options,
            title=title,
            size=(600, 400),
            max_len=100
        )

        # Blend all components
        # blended = grid + figure + labels + title_img
        blended = jplt.blend(grid, figure, labels, title_img)

        # Convert to uint8 for display
        blended = blended.astype(np.uint8)

        # Display the result
        cv2.imshow("Plot Test", cv2.cvtColor(blended, cv2.COLOR_RGBA2BGR))

        # Check for key press
        key = cv2.waitKey(30)
        if key == 27:  # ESC key
            break
        elif key == 32:  # Spacebar key
            cv2.waitKey(0)  # Wait indefinitely until another key is pressed

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_plot() 
import numpy as np
import cv2
from justpyplot.justpyplot import plot1_components, plot

def dumped_sine_wave(N, t):
    x = np.linspace(0+0.05*t, 4*np.pi+0.05*t, N)
    y = np.sin(x) * np.exp(-x/10)
    return x, y

def dumped_sine_wave_noise(N, t):
    x = np.linspace(0+0.05*t, 4*np.pi+0.05*t, N)
    y = np.sin(x) * np.exp(-x/10) + np.random.normal(0, 0.1, N)
    return x, y


def simple_test_plot_components():
    # Generate sample data
    x = np.linspace(0, 4*np.pi, 100)
    y = np.sin(x) * np.exp(-x/10)

    # Call plot1_components
    figure, grid, labels, title_img = plot1_components(
        np.vstack((x, y)),
        title="Damped Sine Wave",
        size=(600, 400),
        scatter=False,
    )

    # Simple blending (just add all components)
    blended = figure + grid + labels + title_img

    # Convert to uint8 for display
    blended = blended.astype(np.uint8)
    # Display the result
    cv2.imshow("Simple Damped Sine Wave Plot", cv2.cvtColor(blended, cv2.COLOR_RGBA2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_in_loop(plotfunc, N, *args, **kwargs):
    cv2.namedWindow("Dynamic Plot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dynamic Plot", 600, 400)

    for i in range(N):

        figure, grid, labels, title_img = plotfunc(i, *args, **kwargs)
        blended = grid + labels + title_img + figure
        overlay = np.where((grid>0) & (figure>0))
        blended[overlay] = figure[overlay]
        cv2.imshow("Dynamic Plot", cv2.cvtColor(blended, cv2.COLOR_RGBA2BGR))

        # Toggle pause state if space is pressed, quiton ESC
        key = cv2.waitKey(30)
        if key == 32: #Space key
            key = cv2.waitKey(0)
            if key == 32:
                continue
            if key == 27:
                break
        elif key == 27:  # ESC key
            break
        

    cv2.destroyAllWindows()
        
def test_plot_components_loop():
    # Create a window
    cv2.namedWindow("Dynamic Plot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dynamic Plot", 600, 400)

    for i in range(500):
        # Generate sample data
        x = np.linspace(0, 4*np.pi, 100)
        y = np.sin(x + i*0.1) * np.exp(-x/10)

        # Call plot1_components
        figure, grid, labels, title_img = plot1_components(
            np.vstack((x, y)),
            title=f"Dynamic Wave (Frame {i})",
            size=(600, 400),
            scatter=False,
        )

        # Simple blending (just add all components)
        blended = figure + grid + labels + title_img

        # Convert to uint8 for display
        blended = blended.astype(np.uint8)
        # Display the result
        cv2.imshow("Dynamic Plot", cv2.cvtColor(blended, cv2.COLOR_RGBA2BGR))
        # Break the loop if 'ESC' is pressed
        if cv2.waitKey(30) == 27:
            break

    cv2.destroyAllWindows()

def scatter_plot(i, *args, **kwargs):
    x = np.linspace(i*0.1, i*0.1+4*np.pi, 50)
    y = np.sin(x) * np.exp(-x/10) + np.random.normal(0, 0.1, 50)

    figure, grid, labels, title_img = plot1_components(
        np.vstack((x, y)),
        title=f"Scatter Plot (Frame {i})",
        size=(600, 400),
        scatter=True,
        point_color=(255, 0, 0, 255),  # Red points
        r=3,  # Larger points
        *args,
        **kwargs
    )

    return figure, grid, labels, title_img

def line_plot_with_custom_colors(i, *args, **kwargs):
    x = np.linspace(i*0.1, i*0.1+4*np.pi, 100)
    y = np.sin(x) * np.exp(-x/10)

    figure, grid, labels, title_img = plot1_components(
        np.vstack((x, y)),
        title=f"Custom Color Line Plot (Frame {i})",
        size=(600, 400),
        scatter=False,
        line_color=(0, 255, 0, 255),  # Green line
        grid_color=(200, 200, 200, 128),  # Light gray, semi-transparent grid
        label_color=(255, 0, 0, 255),  # Red labels
        *args,
        **kwargs
    )

    return figure,grid,labels,title_img

def high_precision_plot(i):
    x = np.linspace(i*0.1, i*0.1+1, 100)
    y = x*np.sin(x) * 0.1

    figure, grid, labels, title_img = plot1_components(
        np.vstack((x, y)),
        title=f"High Precision Plot (Frame {i})",
        size=(800, 600),
        scatter=False,
        precision=6,  # Higher precision for y-axis labels
        nticks=20,  # More ticks
    )

    return figure,grid,labels,title_img

def thick_line_plot(i, *args, **kwargs):
    x = np.linspace(0, 4*np.pi, 100)
    y = np.sin(x + i*0.1)

    figure, grid, labels, title_img = plot1_components(
        np.vstack((x, y)),
        title=f"Thick Line Plot (Frame {i})",
        size=(600, 400),
        scatter=False,
        thickness=5,  # Thicker line
        line_color=(255, 165, 0, 255),  # Orange line
        *args,
        **kwargs
    )

    return figure ,grid ,labels , title_img

def plot_dict_line(i, *args, **kwargs):
    x = np.linspace(i*0.1, i*0.1+4*np.pi, 100)
    y = np.sin(x) * np.exp(-x/10)

    grid_style = {
        'color': (200, 200, 200, 128),  # Light gray, semi-transparent grid
        'label_color': (255, 0, 0, 255),  # Red labels
    }

    figure_style = {
        'scatter': False,
        'line_color': (0, 255, 0, 255),  # Green line
    }

    figure, grid, labels, title_img = plot(
        np.vstack((x, y)),
        grid=grid_style,
        figure=figure_style,
        title=f"Dict Line Plot (Frame {i})",
        size=(600, 400),
        *args,
        **kwargs
    )

    return figure, grid, labels, title_img

def plot_dict_scatter(i, *args, **kwargs):
    x = np.linspace(i*0.1, i*0.1+4*np.pi, 50)
    y = np.sin(x) * np.exp(-x/10) + np.random.normal(0, 0.1, 50)

    grid_style = {
        'color': (128, 128, 128, 200),  # Gray, semi-transparent grid
        'label_color': (0, 0, 255, 255),  # Blue labels
        'nticks': 8,
    }

    figure_style = {
        'scatter': True,
        'point_color': (255, 0, 0, 255),  # Red points
        'point_radius': 3,  # Larger points
    }

    figure, grid, labels, title_img = plot(
        np.vstack((x, y)),
        grid=grid_style,
        figure=figure_style,
        title=f"Dict Scatter Plot (Frame {i})",
        size=(600, 400),
        *args,
        **kwargs
    )

    return figure, grid, labels, title_img

def plot_dict_thick_line(i, *args, **kwargs):
    x = np.linspace(0, 4*np.pi, 100)
    y = np.sin(x + i*0.1)

    grid_style = {
        'color': (200, 200, 200, 128),  # Light gray, semi-transparent grid
        'label_color': (0, 0, 0, 255),  # Black labels
        'precision': 3,
    }

    figure_style = {
        'scatter': False,
        'line_color': (255, 165, 0, 255),  # Orange line
        'thickness': 5,  # Thicker line
    }

    figure, grid, labels, title_img = plot(
        np.vstack((x, y)),
        grid=grid_style,
        figure=figure_style,
        title=f"Dict Thick Line Plot (Frame {i})",
        size=(600, 400),
        *args,
        **kwargs
    )

    return figure, grid, labels, title_img

def plot_dict_dashed_line(i, *args, **kwargs):
    x = np.linspace(i*0.1, i*0.1+4*np.pi, 100)
    y = np.sin(x) * np.exp(-x/10)

    grid_style = {
        'color': (200, 200, 200, 128),  # Light gray, semi-transparent grid
        'label_color': (0, 0, 255, 255),  # Blue labels
    }

    figure_style = {
        'scatter': False,
        'line_color': (255, 0, 0, 255),  # Red line
        'line_style': 'dashed',
        'line_thickness': 2,
        'marker_style': 'o_filled',
    }

    figure, grid, labels, title_img = plot(
        np.vstack((x, y)),
        grid=grid_style,
        figure=figure_style,
        title=f"Dict Dashed Line Plot (Frame {i})",
        size=(600, 400),
        *args,
        **kwargs
    )

    return figure, grid, labels, title_img

def plot_dict_dotted_line(i, *args, **kwargs):
    x = np.linspace(i*0.1, i*0.1+4*np.pi, 100)
    y = np.sin(x) * np.exp(-x/10)

    grid_style = {
        'color': (200, 200, 200, 128),  # Light gray, semi-transparent grid
        'label_color': (0, 255, 0, 255),  # Green labels
    }

    figure_style = {
        'scatter': False,
        'line_color': (255, 0, 0, 255),  # Blue line
        'line_style': 'dashed',
        'dash_length': 10,
        'line_thickness': 1,
        'marker_style': 'triangle',
        'point_color': (0, 255, 0, 255),
        'point_radius': 4,
    }

    figure, grid, labels, title_img = plot(
        np.vstack((x, y)),
        grid=grid_style,
        figure=figure_style,
        title=f"Dict Dotted Line Plot (Frame {i})",
        size=(600, 400),
        *args,
        **kwargs
    )

    return figure, grid, labels, title_img

def plot_dict_cross_markers(i, *args, **kwargs):
    x = np.linspace(i*0.1, i*0.1+4*np.pi, 50)
    y = np.sin(x) * np.exp(-x/10) + np.random.normal(0, 0.1, 50)

    grid_style = {
        'color': (128, 128, 128, 200),  # Gray, semi-transparent grid
        'label_color': (255, 0, 255, 255),  # Magenta labels
        'nticks': 8,
    }

    figure_style = {
        'scatter': True,
        'point_color': (0, 255, 255, 255),  # Cyan points
        'point_radius': 4,  # Larger points
        'marker_style': 'cross',
    }

    figure, grid, labels, title_img = plot(
        np.vstack((x, y)),
        grid=grid_style,
        figure=figure_style,
        title=f"Dict Cross Markers Plot (Frame {i})",
        size=(600, 400),
        *args,
        **kwargs
    )

    return figure, grid, labels, title_img

def plot_dict_square_markers(i, *args, **kwargs):
    x = np.linspace(i*0.1, i*0.1+4*np.pi, 50)
    y = np.sin(x) * np.exp(-x/10) + np.random.normal(0, 0.1, 50)

    grid_style = {
        'color': (128, 128, 128, 200),  # Gray, semi-transparent grid
        'label_color': (255, 165, 0, 255),  # Orange labels
        'nticks': 10,
    }

    figure_style = {
        'scatter': True,
        'point_color': (128, 0, 128, 255),  # Purple points
        'point_radius': 3,  # Larger points
        'marker_style': 'square',
    }

    figure, grid, labels, title_img = plot(
        np.vstack((x, y)),
        grid=grid_style,
        figure=figure_style,
        title=f"Dict Square Markers Plot (Frame {i})",
        size=(600, 400),
        *args,
        **kwargs
    )

    return figure, grid, labels, title_img


if __name__ == "__main__":
    # print("Running simple test...")
    # simple_test_plot_components()
    
    print("Running loop test...")
    test_in_loop(plot_dict_dotted_line, N=500, bounds=np.array([[None, None],[-1, 1]]))

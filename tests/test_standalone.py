import numpy as np
import pyvista as pv
from justpyplot import justpyplot as jplt
import time

# Create plotter
plotter = pv.Plotter()
plotter.add_axes()

def update_plot(phase=0):
    # Create data
    x = np.linspace(-10, 10, 100)
    z = np.sin(x + phase)
    plot_data = np.array([x, z])

    # Create 2D plot using jplt
    plot_array = jplt.plot(
        plot_data,
        grid={'nticks': 5, 'color': (128, 128, 128, 255)},
        figure={'scatter': False, 'line_color': (255, 0, 0, 255), 'line_width': 2},
        title='Sine Wave',
        size=(400, 300)
    )
    blended = jplt.blend(*plot_array)

    # Create a surface by rotating the sine wave
    theta = np.linspace(0, 2*np.pi, 100)
    x_grid, theta_grid = np.meshgrid(x, theta)
    r = z  # use sine wave values as radius
    y_grid = r * np.cos(theta_grid)
    z_grid = r * np.sin(theta_grid)

    # Create PyVista structured grid
    grid = pv.StructuredGrid(x_grid, y_grid, z_grid)

    # Map texture to plane
    grid.texture_map_to_plane()
    grid.active_texture = pv.numpy_to_texture(blended)

    return grid

# Set up the plotter for animation
plotter.open_movie('animation.mp4', framerate=24)
mesh = None
phase = 0

# Update every frame
for i in range(120):  # More frames for smoother animation
    if mesh:
        plotter.remove_actor(mesh)
    
    grid = update_plot(phase)
    mesh = plotter.add_mesh(grid)
    
    # Rotate camera
    plotter.camera.azimuth = i * 3  # Rotate 3 degrees per frame
    plotter.camera.elevation = 20  # Fixed elevation angle
    
    plotter.write_frame()
    phase += 0.1  # Smaller phase increment for smoother wave motion
    
    # No sleep for smoother animation
    
# Close the plotter
plotter.close()
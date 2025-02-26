import numpy as np 
import cv2
import time
from justpyplot import justpyplot as jplt

t0 = time.perf_counter()
xs, ys = [], []

while(cv2.waitKey(1) != 27):  # Press ESC to exit
    xt = time.perf_counter() - t0
    yx = np.sin(xt)
    xs.append(xt)
    ys.append(yx)
    
    # Create plot data array
    plot_data = np.array([xs, ys])
    
    # Generate plot arrays with customized options
    figure, grid, labels, title_img = jplt.plot(
        plot_data,
        grid={'nticks': 5, 'color': (128, 128, 128, 255)},
        figure={'scatter': True, 'line_color': (255, 0, 0, 255), 'line_width': 2},
        title="sin() from Clock",
        size=(600, 400)
    )
    
    # Blend all plot layers into single image
    plotted_array = jplt.blend(figure, grid, labels, title_img)
    
    cv2.imshow('np array plot', plotted_array)
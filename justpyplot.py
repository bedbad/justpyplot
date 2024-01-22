# Copyright (c) 2023 bedbad
import numpy as np
from typing import Tuple, List

import cv2
from perf_timer import PerfTimer


_veclinesperf = PerfTimer('vectorized lines render')


def vectorized_line(y0, x0, y1, x1, canvas_size, thickness):
    # Create an array of distances
    num_points = max(np.max(abs(x1 - x0)), np.max(abs(y1 - y0))) + 1
    t = np.linspace(0, 1, num_points)
    # Create 2D arrays for x and y coordinates
    x = (x0 + np.outer(t, (x1 - x0))).astype(int)
    y = (y0 + np.outer(t, (y1 - y0))).astype(int)
    # Create a boolean mask with the size of the canvas and an additional dimension for t
    mask = np.zeros(canvas_size, dtype=bool)
    # Set the corresponding positions to True
    mask[y.ravel(), x.ravel()] = True
    return mask

def vectorized_lines(y0, x0, y1, x1, img_array, clr=(0,0,255)):
    # Create an array of distances
    num_points = max(np.max(abs(x1 - x0)), np.max(abs(y1 - y0))) + 1
    t = np.linspace(0, 1, num_points)
    # Create 2D arrays for x and y coordinates
    x = (x0 + np.outer(t, (x1 - x0))).astype(int)
    y = (y0 + np.outer(t, (y1 - y0))).astype(int)
    # Set the corresponding positions to clr
    img_array[y.ravel(), x.ravel()] = clr
    return img_array

def vectorized_lines_with_thickness(y0, x0, y1, x1, img_array, thickness, clr=(0, 0, 255)):
    """
    Draw multiple lines with specified thickness on an image array.

    This function uses vectorized operations to draw lines between pairs of points
    defined by corresponding elements in the x0, y0 (start points) and x1, y1 (end points)
    arrays. It modifies the input image array in-place by setting the color of the pixels
    along the lines to the specified color.

    Parameters:
    y0 (np.ndarray): An array of y-coordinates for the start points of the lines.
    x0 (np.ndarray): An array of x-coordinates for the start points of the lines.
    y1 (np.ndarray): An array of y-coordinates for the end points of the lines.
    x1 (np.ndarray): An array of x-coordinates for the end points of the lines.
    img_array (np.ndarray): The image array on which to draw the lines. This array will be modified in-place.
    thickness (int): The thickness of the lines to be drawn.
    clr (tuple): A tuple of three integers representing the color of the lines in BGR (blue, green, red) format.

    Returns:
    np.ndarray: The modified image array with the lines drawn on it.

    Example:
    >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
    >>> y0 = np.array([10, 20])
    >>> x0 = np.array([10, 20])
    >>> y1 = np.array([80, 80])
    >>> x1 = np.array([80, 30])
    >>> vectorized_lines_with_thickness(y0, x0, y1, x1, img, 3, (255, 0, 0))
    """
    # Create an array of distances
    num_points = max(np.max(abs(x1 - x0)), np.max(abs(y1 - y0))) + 1
    t = np.linspace(0, 1, num_points)
    # Create 2D arrays for x and y coordinates
    x = (x0 + np.outer(t, (x1 - x0))).astype(int)
    y = (y0 + np.outer(t, (y1 - y0))).astype(int)
    # Create the shift indices
    shift_indices = np.arange(-thickness//2, thickness//2 + 1)
    # Ensure that the shift is broadcastable by adding a new axis to y1 and y0
    y1 = y1[:, np.newaxis]
    y0 = y0[:, np.newaxis]
    x1 = x1[:, np.newaxis]
    x0 = x0[:, np.newaxis]
    # Create the shifted coordinates
    x_shifted = x[..., np.newaxis] + shift_indices * np.sign(x1 - x0)
    y_shifted = y[..., np.newaxis] + shift_indices * np.sign(y1 - y0)
    # Clip the shifted coordinates to the image boundaries
    x_shifted = np.clip(x_shifted, 0, img_array.shape[1] - 1)
    y_shifted = np.clip(y_shifted, 0, img_array.shape[0] - 1)
    # Flatten the arrays to set the color in the image array
    img_array[y_shifted.ravel(), x_shifted.ravel()] = clr
    return img_array

def just_plot(img_array: np.array, 
                       values: np.array,
                       title: str = "Measuring", 
                       offset: Tuple[int, int] = (50, 50), 
                       size: Tuple[int, int] = (300,270), 
                       clr: Tuple[int, int, int] = (0, 0, 255), 
                       r = 2,
                       pxdelta: int = 15, 
                       precision: int = 2, 
                       default_font_size: float = .75, 
                       default_font_size_small: float = .5,
                       scatter = False,
                       thickness = 2,
                       opacity: float = 1.0,
                       max_len: int = 100) -> np.array:
    """Adaptively draw a plot on a NumPy image array.

        Plots given `values` to a given NumPy ndarray, adapting
        the plot scale and size to fit the input data.
        Plots fast - no single loop in the code, even if you want to connect points with
        line segments, measured 20-100x faster then matplotlib.
        Useful for overlaying real-time plots on images and video frames.

        Args:
            img_array: NumPy ndarray to draw the plot on, likely a video frame
            values: NumPy 1D array of values to plot over time
            title: Plot title string  
            offset: (x, y) offset tuple for the top-left of plot 
            size: (width, height) tuple for plot size in pixels
            clr: (R, G, B) tuple for plot color
            pxdelta: Grid size in pixels 
            precision: Floating point precision for y-axis labels  
            default_font_size: Font size for title   
            default_font_size_small: Font size for axis labels
            opacity: Opacity value 0-1 for plot elements
            max_len: Maximum history length for values array

        Returns:
            img_array: Image array with overlaid adaptive plot

        Example:
            frame = cv2.imread('frame.jpg') 
            values = sensor_data[-100:]
            frame = draw_adaptive_plot(frame, values)
        """
    min_val = np.min(values)
    max_val = np.max(values)
    
    if max_len > 0:
        values = values[-max_len:]

    # Calculate adjustment factor and shift
    if max_val - min_val == 0:
        if min_val == 0:
            scale = 1.
            shift = -.5
            power = 1
            adjust_factor = 1
        else:
            scale = min_val / 2
            power = np.floor(np.log10(min_val))
            adjust_factor = 1
            shift = -scale
    else:
        scale = (max_val - min_val)
        power = np.ceil(np.log10((np.abs(min_val)+np.abs(max_val))/2))
        adjust_factor = size[1] / scale
        shift = -min_val

        # Determine the multiplier to scale the tick values above 1.0
    
    multiplier = 10 ** -power

    # Adjust the title to include the multiplier

    title += f", 10^{int(power)}"

    # Adjust values
    adjusted_values = (values + shift) * adjust_factor

    # Draw the plot
    height = size[1]
    width = size[0]

    top_left = (offset[0], offset[1])
    bottom_right = (offset[0] + width, offset[1] + height)

    font_size = default_font_size
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw grid and rectangle with opacity
    grid_color = (128, 128, 128)
    img_array[top_left[1]:bottom_right[1]+1:pxdelta, top_left[0]:bottom_right[0]+1,:] = grid_color
    img_array[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1:pxdelta,:] = grid_color

    # Render points 
    point_color = clr

    # Create an array of indices
    i = np.arange(len(adjusted_values))
    x = bottom_right[0] - ((i + 1) * width // len(adjusted_values))
    y = bottom_right[1] - (adjusted_values).astype(int)

    # Create a mask for valid indices
    valid_mask = (top_left[0] <= x) & (x <= bottom_right[0] ) & \
                (top_left[1] <= y) & (y <= bottom_right[1])
    valsx = x[valid_mask]
    valsy = y[valid_mask]
    # Create a grid of offsets
    x_offset = np.arange(-r, r + 1)
    y_offset = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(x_offset, y_offset)

    # Apply offsets to the original x and y coordinates
    xx = xx.ravel() + valsx[:, None]
    yy = yy.ravel() + valsy[:, None]

    # Flatten the arrays
    xx = xx.ravel()
    yy = yy.ravel()

    # Assign color to the corresponding pixels and the surrounding pixels
    img_array[yy, xx] = point_color

    if (not scatter and values.shape[0]>=2):
        # Create pairs of adjacent points
        with _veclinesperf:
            img_array = vectorized_lines_with_thickness(y[:-1], x[:-1], y[1:], x[1:],img_array,clr=(255,0,0), thickness=thickness)

    #rendering text
    font_size_small = default_font_size_small
    n = height // (2 * pxdelta)
    tick_color = clr
    for i in range(n + 1):
        # Scale the tick label by the multiplier
        val = "{:.{}f}".format((scale / n * i) * multiplier, precision)
        text_size, _ = cv2.getTextSize(val, font, font_size_small, 1)
        text_width, text_height = text_size
        text_x = top_left[0] - text_width - 5  # Adjust position to the left of the grid
        text_y = bottom_right[1] - i * 2 * pxdelta + pxdelta // 2
        cv2.putText(img_array, val, (text_x, text_y), font, font_size_small, tick_color, 1)

    # Draw title with opacity
    title_color = clr
    text_size_title = cv2.getTextSize(title, font, font_size, 1)[0]
    text_x_title = top_left[0] + width // 2 - text_size_title[0] // 2
    text_y_title = top_left[1] - text_size_title[1] - pxdelta // 2
    cv2.putText(img_array, title, (text_x_title, text_y_title), font, font_size, title_color, 1)

    return img_array
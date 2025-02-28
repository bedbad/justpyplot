# Copyright (c) 2023 bedbad
import numpy as np
from typing import Tuple, Optional
import functools
import importlib
from justpyplot.textrender import vectorized_text

__all__ = [
    'plot',       # Main plotting function
    'blend',      # Core blending function for numpy arrays
    'blend2PIL',  # Specialized blending for Jupyter/PIL output
    'plot_at',     # Plot directly onto existing array
    'plot1_at',   # Plot 1D array onto existing array
]


# Attempt to import optional modules
def is_module_available(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

cv2_available = is_module_available("cv2")
perf_timer_available = is_module_available("perf_timer")
PIL_available = is_module_available("PIL")

if cv2_available:
    import cv2

if perf_timer_available:
    from perf_timer import PerfTimer
    perf_timers = {
        '_veclinesperf': PerfTimer('vectorized lines render'),
        '_plotperf': PerfTimer('full plot rendering')
    }
else:
    perf_timers = {}


def debug_performance(perf_name: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if perf_name in perf_timers:
                with perf_timers[perf_name]:
                    result = func(*args, **kwargs)
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def adjust_values(values, grid_shape):
    """
    Adjusts the values to fill the grid box maximally.

    Parameters:
    values (ndarray): The input array of values.
    grid_shape (tuple): The shape of the grid box.

    Returns:
    ndarray: The adjusted values.
    ndarray: The bounds of the values.
    ndarray: The scaling factor.
    ndarray: The median degree of the values.
    """
    # Calculate the bounds for both rows of the array values
    bounds = np.array([np.min(values, axis=1), np.max(values, axis=1)])

    # Calculate the range of the values
    value_range = bounds[1] - bounds[0]
    value_range[value_range == 0] = 1

    # Calculate the scaling factor
    scale = np.array(grid_shape) / value_range

    # Adjust the values to fill the grid box maximally
    adjusted_values = (values - bounds[0, :, np.newaxis]) * scale[:, np.newaxis]

    # Calculate the median degree for values in both rows and round to the nearest whole number
    median_degree = np.round(np.median(values, axis=1)).astype(int)

    return adjusted_values, bounds, scale, median_degree


def adjust_values_maxlen(values, grid_shape, max_len):
    """
    Adjusts the values array to a maximum length and scales it to fit a grid box.

    Parameters:
    values (ndarray): The input array of values.
    grid_shape (tuple): The shape of the grid box.
    max_len (int): The maximum length of the values array.

    Returns:
    ndarray: The adjusted values array.
    ndarray: The bounds of the values array.
    ndarray: The scaling factor.
    ndarray: The median degree of the values array.
    """
    # Calculate the bounds for both rows of the array values

    ybounds = np.array([np.min(values[1, :]), np.max(values[1, :])])
    values = values[:, -max_len:]
    xbounds = np.array([np.min(values[0]), np.max(values[0])])
    bounds = np.stack([xbounds, ybounds], axis=1)
    # Calculate the range of the values
    value_range = np.array([xbounds[1] - xbounds[0], ybounds[1] - ybounds[0]])
    median_degree = np.array([0, 0])
    if value_range[0] == 0:
        value_range[0] = values[0]
        median_degree[0] = 0
    if value_range[1] == 0:
        value_range[1] = values[1]
        median_degree[1] = 0
    if value_range[0] and value_range[1]:
        median_degree = np.round(np.log10(np.median(np.abs(values), axis=1))).astype(
            int
        )

    # Calculate the scaling factor
    scale = np.array(grid_shape[::-1]) / value_range

    # Adjust the values to fill the grid box maximally
    adjusted_values = (values - bounds[0, :, np.newaxis]) * (scale[:, np.newaxis])

    # Calculate the median degree for values in both rows and round to the nearest whole number

    return adjusted_values, bounds, scale, median_degree


def vectorized_line(y0, x0, y1, x1, canvas_size, thickness):
    """
    Generate a boolean mask representing a vectorized line on a canvas.

    Parameters:
    y0 (int): The y-coordinate of the starting point of the line.
    x0 (int): The x-coordinate of the starting point of the line.
    y1 (int): The y-coordinate of the ending point of the line.
    x1 (int): The x-coordinate of the ending point of the line.
    canvas_size (tuple): The size of the canvas as a tuple (height, width).
    thickness (int): The thickness of the line.

    Returns:
    numpy.ndarray: A boolean mask representing the line on the canvas.
    """
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


@debug_performance('_veclinesperf')
def vectorized_lines(y0, x0, y1, x1, img_array, clr=(0, 0, 255)):
    """
    Draw vectorized lines on an image array.

    Parameters:
    y0 (array-like): Starting y-coordinates of the lines.
    x0 (array-like): Starting x-coordinates of the lines.
    y1 (array-like): Ending y-coordinates of the lines.
    x1 (array-like): Ending x-coordinates of the lines.
    img_array (ndarray): Image array on which the lines will be drawn.
    clr (tuple, optional): RGB color tuple for the lines. Defaults to (0, 0, 255).

    Returns:
    ndarray: Image array with the lines drawn.
    """
    # Create an array of distances
    num_points = max(np.max(abs(x1 - x0)), np.max(abs(y1 - y0))) + 1
    t = np.linspace(0, 1, num_points)
    # Create 2D arrays for x and y coordinates
    x = (x0 + np.outer(t, (x1 - x0))).astype(int)
    y = (y0 + np.outer(t, (y1 - y0))).astype(int)
    # Set the corresponding positions to clr
    img_array[y.ravel(), x.ravel()] = clr
    return img_array


@debug_performance('_veclinesperf')
def vectorized_lines_with_thickness(
    y0, x0, y1, x1, img_array, thickness, clr=(0, 0, 255)
):
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
    shift_indices = np.arange(-thickness // 2, thickness // 2 + 1)
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


def plot2_at(
    img_array: np.ndarray,
    values: np.array,
    offset: Tuple[int, int],
    title: str = 'Measuring',
    size: Tuple[int, int] = (300, 300),
    point_color: Tuple[int, int, int, int] = (0, 0, 255),
    r=2,
    nticks: int = 16,
    grid_color: Tuple[int, int, int, int] = (128, 128, 128),
    precision: int = 4,
    default_font_size: float = 0.5,
    default_font_size_small: float = 0.4,
    label_color: Tuple[int, int, int, int] = (0, 0, 255),
    scatter=True,
    thickness=2,
    line_color: Tuple[int, int, int, int] = (0, 0, 255),
    max_len: int = 100,
) -> np.ndarray:
    """Plot into a NumPy image array.

    Plots given array of `values`, adapting
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

    font_size = default_font_size
    font_size_small = default_font_size_small
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size_title = cv2.getTextSize(title, font, font_size, 1)[0]
    margin_ver = int(text_size_title[1] * 2.0)

    axlablen = cv2.getTextSize('A' * precision, font, font_size_small, 1)[0][0]
    margin_hor = int(axlablen * 1.5)

    grid_topleft = np.array((margin_ver, margin_hor))
    grid_botright = np.array(size) - grid_topleft
    gsize = grid_botright - grid_topleft
    gsize2 = gsize - (gsize % nticks)
    iota = (gsize - gsize2) / 2
    gsize = gsize2
    grid_topleft = (grid_topleft + iota).astype(int)
    grid_botright = (grid_botright - iota).astype(int)
    pxdelta = (gsize // nticks).astype(int)

    gh, gw = tuple(gsize)
    adjusted_values, bounds, scale, median_degree = adjust_values_maxlen(
        values, gsize, max_len=max_len
    )
    pxdelta = (gsize // nticks).astype(int)
    # Adjust the title to include the multiplier

    # Draw grid and rectangle with opacity
    gtl_img = grid_topleft + offset
    gbr_img = grid_botright + offset

    title += f', 10^{int(median_degree[1])}'
    text_x_title = int(
        gtl_img[1] + gw / 2 - cv2.getTextSize(title, font, font_size, 1)[0][0] / 2
    )
    text_y_title = gtl_img[0] - int(text_size_title[1] * 1.5)

    img_array[
        gtl_img[0] : gbr_img[0] + 1 : pxdelta[0], gtl_img[1] : gbr_img[1] + 1, :
    ] = grid_color
    img_array[
        gtl_img[0] : gbr_img[0] + 1, gtl_img[1] : gbr_img[1] + 1 : pxdelta[1], :
    ] = grid_color

    # Render points

    # Create an array of indices
    x = gtl_img[1] + (adjusted_values[0, ...]).astype(int)
    y = gbr_img[0] - (adjusted_values[1, ...]).astype(int)

    # Create a mask for valid indices
    valid_mask = (
        (gtl_img[0] <= y) & (y <= gbr_img[0]) & (gtl_img[1] <= x) & (x <= gbr_img[1])
    )
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

    if not scatter and values.shape[1] >= 2:
        # Create pairs of adjacent points
        with _veclinesperf:
            img_array = vectorized_lines_with_thickness(
                y[:-1],
                x[:-1],
                y[1:],
                x[1:],
                img_array,
                clr=line_color,
                thickness=thickness,
            )

    # rendering text
    n = gsize[0] // (2 * pxdelta[0])
    tick_color = label_color
    yscale = bounds[1, 1] - bounds[0, 1]
    for i in range(n + 1):
        # Scale the tick label by the multiplier
        tickval = bounds[0, 1] + (yscale / n) * i
        dotp = precision - len(str(tickval).split('.')[0])
        val = '{:.{}f}'.format(tickval, dotp)
        text_size, _ = cv2.getTextSize(val, font, font_size_small, 1)
        text_width, text_height = text_size
        text_x = offset[1] + pxdelta[1] // 2  # Adjust position to the left of the grid
        text_y = gbr_img[0] - i * 2 * pxdelta[0] + text_height // 2
        cv2.putText(
            img_array, val, (text_x, text_y), font, font_size_small, tick_color, 1
        )

    # Draw title with opacity
    cv2.putText(
        img_array, title, (text_x_title, text_y_title), font, font_size, label_color, 1
    )
    return img_array


@debug_performance('_plotperf')
def plot2(
    values: np.array,
    title: str = 'Measuring',
    size: Tuple[int, int] = (300, 300),
    point_color: Tuple[int, int, int, int] = (0, 0, 255, 255),
    r=2,
    nticks: int = 16,
    grid_color: Tuple[int, int, int, int] = (128, 128, 128, 255),
    precision: int = 4,
    default_font_size: float = 0.5,
    default_font_size_small: float = 0.4,
    label_color: Tuple[int, int, int, int] = (0, 0, 255, 255),
    scatter=True,
    thickness=2,
    line_color: Tuple[int, int, int, int] = (0, 0, 255, 255),
    max_len: int = 100,
) -> np.array:
    """Plot into a NumPy image array.

    Plots given array of `values`, adapting
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

    font_size = default_font_size
    font_size_small = default_font_size_small
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size_title = cv2.getTextSize(title, font, font_size, 1)[0]
    text_y_title = int(text_size_title[1] * 1.5)
    margin_ver = int(text_y_title + text_size_title[1] * 0.5)

    axlablen = cv2.getTextSize('A' * precision, font, font_size_small, 1)[0][0]
    margin_hor = int(axlablen * 1.5)

    grid_topleft = np.array((margin_ver, margin_hor))
    grid_botright = np.array(size) - grid_topleft
    gsize = grid_botright - grid_topleft
    gsize2 = gsize - (gsize % nticks)
    iota = (gsize - gsize2) / 2
    gsize = gsize2
    grid_topleft = (grid_topleft + iota).astype(int)
    grid_botright = (grid_botright - iota).astype(int)
    pxdelta = (gsize // nticks).astype(int)

    gh, gw = tuple(gsize)
    adjusted_values, bounds, scale, median_degree = adjust_values_maxlen(
        values, gsize, max_len=max_len
    )
    title += f', 10^{int(median_degree[1])}'
    text_x_title = int(
        grid_topleft[1] + gw / 2 - cv2.getTextSize(title, font, font_size, 1)[0][0] / 2
    )
    pxdelta = (gsize // nticks).astype(int)

    img_array = np.zeros((*size, 4), np.uint8)
    # Adjust the title to include the multiplier

    # Draw grid and rectangle with opacity
    img_array[
        grid_topleft[0] : grid_botright[0] + 1 : pxdelta[0],
        grid_topleft[1] : grid_botright[1] + 1,
        :,
    ] = grid_color
    img_array[
        grid_topleft[0] : grid_botright[0] + 1,
        grid_topleft[1] : grid_botright[1] + 1 : pxdelta[1],
        :,
    ] = grid_color

    # Render points

    # Create an array of indices
    x = grid_topleft[1] + (adjusted_values[0, ...]).astype(int)
    y = grid_botright[0] - (adjusted_values[1, ...]).astype(int)

    # Create a mask for valid indices
    valid_mask = (
        (grid_topleft[0] <= y)
        & (y <= grid_botright[0])
        & (grid_topleft[1] <= x)
        & (x <= grid_botright[1])
    )
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

    if not scatter and values.shape[1] >= 2:
        # Create pairs of adjacent points
        with _veclinesperf:
            img_array = vectorized_lines_with_thickness(
                y[:-1],
                x[:-1],
                y[1:],
                x[1:],
                img_array,
                clr=line_color,
                thickness=thickness,
            )

    # rendering text
    n = gsize[0] // (2 * pxdelta[0])
    tick_color = label_color
    yscale = bounds[1, 1] - bounds[0, 1]
    for i in range(n + 1):
        # Scale the tick label by the multiplier
        tickval = bounds[0, 1] + (yscale / n) * i
        dotp = precision - len(str(tickval).split('.')[0])
        val = '{:.{}f}'.format(tickval, dotp)
        text_size, _ = cv2.getTextSize(val, font, font_size_small, 1)
        text_width, text_height = text_size
        text_x = pxdelta[1] // 2  # Adjust position to the left of the grid
        text_y = grid_botright[0] - i * 2 * pxdelta[0] + text_height // 2
        cv2.putText(
            img_array, val, (text_x, text_y), font, font_size_small, tick_color, 1
        )

    # Draw title with opacity
    cv2.putText(
        img_array, title, (text_x_title, text_y_title), font, font_size, label_color, 1
    )
    return img_array


@debug_performance('_plotperf')
def plot1_cv(
    values: np.array,
    title: str = 'Measuring',
    size: Tuple[int, int] = (300, 300),
    point_color: Tuple[int, int, int, int] = (0, 0, 255, 255),
    r=2,
    nticks: int = 16,
    grid_color: Tuple[int, int, int, int] = (128, 128, 128, 255),
    precision: int = 2,
    default_font_size: float = 0.5,
    default_font_size_small: float = 0.4,
    label_color: Tuple[int, int, int, int] = (0, 0, 255, 255),
    scatter=True,
    thickness=2,
    line_color: Tuple[int, int, int, int] = (0, 0, 255, 255),
    max_len: int = 100,
) -> np.array:
    """Plot into a NumPy image array.

    Plots given array of `values`, adapting
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
            scale = 1.0
            shift = -0.5
            power = 1
            adjust_factor = 1
        else:
            scale = min_val / 2
            power = np.floor(np.log10(min_val))
            adjust_factor = 1
            shift = -scale
    else:
        scale = max_val - min_val
        power = np.ceil(np.log10((np.abs(min_val) + np.abs(max_val)) / 2))
        shift = -min_val

        # Determine the multiplier to scale the tick values above 1.0

    multiplier = 10**-power

    title += f', 10^{int(power)}'
    # Set the paramerics
    # height = size[1]
    # width = size[0]

    font_size = default_font_size
    font_size_small = default_font_size_small
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size_title = cv2.getTextSize(title, font, font_size, 1)[0]
    text_y_title = int(text_size_title[1] * 1.5)
    margin_ver = int(text_y_title + text_size_title[1] * 0.5)

    axlablen = cv2.getTextSize('A' * precision, font, font_size_small, 1)[0][0]
    margin_hor = int(axlablen * 1.5)

    grid_topleft = np.array((margin_ver, margin_hor))
    grid_botright = np.array(size) - grid_topleft
    gsize = grid_botright - grid_topleft
    gsize2 = gsize - (gsize % nticks)
    iota = (gsize - gsize2) / 2
    grid_topleft = (grid_topleft + iota).astype(int)
    grid_botright = (grid_botright - iota).astype(int)
    pxdelta = (gsize // nticks).astype(int)
    gsize = gsize2

    gh, gw = tuple(gsize)
    text_x_title = int(grid_topleft[1] + gw / 2 - text_size_title[0] / 2)
    pxdelta = (gsize // nticks).astype(int)

    img_array = np.zeros((*size, 4), np.uint8)
    # Adjust the title to include the multiplier

    adjust_factor = gsize[0] / scale
    # Adjust values
    adjusted_values = (values + shift) * adjust_factor

    # top_left = (0, 0)
    # bottom_right = (height, width)

    # Draw grid and rectangle with opacity
    img_array[
        grid_topleft[0] : grid_botright[0] + 1 : pxdelta[0],
        grid_topleft[1] : grid_botright[1] + 1,
        :,
    ] = grid_color
    img_array[
        grid_topleft[0] : grid_botright[0] + 1,
        grid_topleft[1] : grid_botright[1] + 1 : pxdelta[1],
        :,
    ] = grid_color

    # Render points

    # Create an array of indices
    i = np.arange(len(adjusted_values))
    x = grid_botright[1] - ((i + 1) * gw // len(adjusted_values))
    y = grid_botright[0] - (adjusted_values).astype(int)

    # Create a mask for valid indices
    valid_mask = (
        (grid_topleft[0] <= y)
        & (y <= grid_botright[0])
        & (grid_topleft[1] <= x)
        & (x <= grid_botright[1])
    )
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

    if not scatter and values.shape[0] >= 2:
        # Create pairs of adjacent points
        with _veclinesperf:
            img_array = vectorized_lines_with_thickness(
                y[:-1],
                x[:-1],
                y[1:],
                x[1:],
                img_array,
                clr=line_color,
                thickness=thickness,
            )

    # rendering text
    n = gsize[0] // (2 * pxdelta[0])
    tick_color = label_color
    for i in range(n + 1):
        # Scale the tick label by the multiplier
        val = '{:.{}f}'.format((scale / n * i) * multiplier, precision)
        text_size, _ = cv2.getTextSize(val, font, font_size_small, 1)
        text_width, text_height = text_size
        text_x = pxdelta[1] // 2  # Adjust position to the left of the grid
        text_y = grid_botright[0] - i * 2 * pxdelta[0] + text_height // 2
        cv2.putText(
            img_array, val, (text_x, text_y), font, font_size_small, tick_color, 1
        )

    # Draw title with opacity
    cv2.putText(
        img_array, title, (text_x_title, text_y_title), font, font_size, label_color, 1
    )
    return img_array

@debug_performance('_plotperf')
def plot1(
    values: np.array,
    title: str = 'Measuring',
    size: Tuple[int, int] = (300, 300),
    point_color: Tuple[int, int, int, int] = (0, 0, 255, 255),
    r=2,
    nticks: int = 16,
    grid_color: Tuple[int, int, int, int] = (128, 128, 128, 255),
    precision: int = 2,
    default_font_size: float = 0.8,
    default_font_size_small: float = 0.6,
    label_color: Tuple[int, int, int, int] = (0, 0, 255, 255),
    scatter=True,
    thickness=2,
    line_color: Tuple[int, int, int, int] = (0, 0, 255, 255),
    max_len: int = 100,
) -> np.array:
    """Draw a plot on a new NumPy image array using textrender for text rendering.

    Creates a new NumPy ndarray and plots the given `values` on it,
    adapting the plot scale and size to fit the input data.
    Plots fast - no single loop in the code, even if you want to connect points with
    line segments, measured 20-100x faster then matplotlib.
    Useful for creating standalone plot images.

    Args:
        values: NumPy 1D array of values to plot over time
        title: Plot title string
        size: (width, height) tuple for plot size in pixels
        point_color: (R, G, B, A) tuple for plot color
        r: Radius of points
        nticks: Number of ticks on the y-axis
        grid_color: (R, G, B, A) tuple for grid color
        precision: Floating point precision for y-axis labels
        default_font_size: Font size for title
        default_font_size_small: Font size for axis labels
        label_color: (R, G, B, A) tuple for label color
        scatter: If True, plot points without connecting lines
        thickness: Thickness of connecting lines
        line_color: (R, G, B, A) tuple for line color
        max_len: Maximum history length for values array

    Returns:
        img_array: New image array with plot

    Example:
        values = sensor_data[-100:]
        plot_img = plot1(values, title="Sensor Data")
    """
    if max_len > 0:
        values = values[-max_len:]
    
    min_val = np.min(values)
    max_val = np.max(values)

    # Calculate adjustment factor and shift
    if max_val - min_val == 0:
        if min_val == 0:
            scale = 1.0
            shift = -0.5
            power = 1
            adjust_factor = 1
        else:
            scale = min_val / 2
            power = np.floor(np.log10(min_val))
            adjust_factor = 1
            shift = -scale
    else:
        scale = max_val - min_val
        power = np.ceil(np.log10((np.abs(min_val) + np.abs(max_val)) / 2))
        shift = -min_val

    multiplier = 10**-power

    title += f', 10^{int(power)}'

    # Estimate text sizes and positions
    font_size = default_font_size  # Adjust this factor as needed
    font_size_small = default_font_size_small  # Adjust this factor as needed
    
    # Estimate margins and grid size
    margin_ver = int(size[1] * 0.1)  # 10% of height for vertical margin
    margin_hor = int(size[0] * 0.15)  # 15% of width for horizontal margin
    grid_topleft = np.array((margin_hor, margin_ver))
    grid_botright = np.array(size) - grid_topleft
    gsize = grid_botright - grid_topleft
    
    # Adjust grid size to be divisible by nticks
    gsize2 = gsize - (gsize % nticks)
    iota = (gsize - gsize2) / 2
    grid_topleft = (grid_topleft + iota).astype(int)
    grid_botright = (grid_botright - iota).astype(int)
    gsize = gsize2
    
    pxdelta = (gsize // nticks).astype(int)

    # Create image array
    img_array = np.zeros((*size, 4), np.uint8)

    adjust_factor = gsize[0] / scale
    adjusted_values = (values + shift) * adjust_factor

    # Draw grid and rectangle with opacity
    img_array[
        grid_topleft[0] : grid_botright[0] + 1 : pxdelta[0],
        grid_topleft[1] : grid_botright[1] + 1,
        :,
    ] = grid_color
    img_array[
        grid_topleft[0] : grid_botright[0] + 1,
        grid_topleft[1] : grid_botright[1] + 1 : pxdelta[1],
        :,
    ] = grid_color

    # Render points
    i = np.arange(len(adjusted_values))
    x = grid_botright[1] - ((i + 1) * gsize[1] // len(adjusted_values))
    y = grid_botright[0] - (adjusted_values).astype(int)

    valid_mask = (
        (grid_topleft[0] <= y)
        & (y <= grid_botright[0])
        & (grid_topleft[1] <= x)
        & (x <= grid_botright[1])
    )
    valsx = x[valid_mask]
    valsy = y[valid_mask]
    x_offset = np.arange(-r, r + 1)
    y_offset = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(x_offset, y_offset)

    xx = xx.ravel() + valsx[:, None]
    yy = yy.ravel() + valsy[:, None]

    xx = xx.ravel()
    yy = yy.ravel()

    img_array[yy, xx] = point_color

    if not scatter and values.shape[0] >= 2:
        with _veclinesperf:
            img_array = vectorized_lines_with_thickness(
                y[:-1],
                x[:-1],
                y[1:],
                x[1:],
                img_array,
                clr=line_color,
                thickness=thickness,
            )

    # Render y-axis labels
    tick_color = label_color[:3]  # Remove alpha channel for vectorized_text
    for i in range(nticks + 1):
        val = '{:.{}f}'.format((scale / nticks * i) * multiplier, precision)
        l = len(val)
        dx = int(l * 5 * font_size_small * 2)
        text_x = grid_topleft[1] - dx  # Approximate text width
        text_y = grid_botright[0] - i * pxdelta[0] - 5 * int(font_size_small)  # Adjust for text height
        img_array = vectorized_text(
            img_array, val, (text_x, text_y), color=tick_color, font_size=font_size_small
        )

    # Draw title
    title_color = label_color[:3]  # Remove alpha channel for vectorized_text
    text_x_title = grid_topleft[1] + (grid_botright[1] - grid_topleft[1]) // 2 - len(title) * 5 * int(font_size * 2) // 2  # Approximate text width
    text_y_title = grid_topleft[0] - int(font_size * 5*2)
    img_array = vectorized_text(
        img_array, title, (text_x_title, text_y_title), color=title_color, font_size=font_size
    )

    return img_array


def blend(*arrays)->np.ndarray:
    """
    Blends multiple NumPy arrays in the order they are provided.
    
    Parameters:
    *arrays: Variable length argument list of NumPy arrays to be blended.
    
    Returns:
    np.ndarray: The blended image if all arrays have the same dimensions,
                otherwise returns the first array.
    """
    if not arrays:
        raise ValueError("At least one array must be provided")

    # Use the first array as the base
    base_array = arrays[0]

    # Check if all arrays have the same shape
    for array in arrays:
        if array.shape != base_array.shape:
            return base_array

    # Blend arrays by overlaying them in order
    blended_array = base_array.copy()
    for array in arrays[1:]:
        alpha = array[..., 3] / 255.0  # Assuming the last channel is alpha
        blended_array[..., :3] = (1 - alpha[..., None]) * blended_array[..., :3] + alpha[..., None] * array[..., :3]

    return blended_array

def blend_at(
    dst_img: np.ndarray, paste_img: np.ndarray, offset: Tuple[int, int]
) -> np.ndarray:
    # pasting image fits
    assert dst_img.ndim == paste_img.ndim and np.all(
        offset + paste_img.shape[0:2] <= dst_img.shape[0:2]
    )
    # it is rgba and proper type
    assert paste_img.shape[2] == 4 and paste_img.dtype == np.uint8
    alpha = paste_img[..., 3][..., None].astype(np.float32) / 255.0
    img = paste_img[..., 0:3]
    sz = img.shape[0:2]
    y0 = offset[0]
    y1 = y0 + sz[0]
    x0 = offset[1]
    x1 = x0 + sz[1]

    dst_img[y0:y1, x0:x1] = dst_img[y0:y1, x0:x1] * (1 - alpha) + img * alpha
    return dst_img


@debug_performance('_plotperf')
def plot1_atcv(
    img_array: np.ndarray,
    values: np.ndarray,
    title: str = 'Measuring',
    offset: Tuple[int, int] = (50, 50),
    size: Tuple[int, int] = (300, 270),
    point_color: Tuple[int, int, int, int] = (0, 0, 255),
    r=2,
    pxdelta: int = 15,
    grid_color: Tuple[int, int, int, int] = (128, 128, 128),
    precision: int = 2,
    default_font_size: float = 0.75,
    default_font_size_small: float = 0.5,
    label_color: Tuple[int, int, int, int] = (0, 0, 255),
    scatter=False,
    thickness=2,
    line_color: Tuple[int, int, int, int] = (0, 0, 255),
    max_len: int = 100,
) -> np.ndarray:
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
            scale = 1.0
            shift = -0.5
            power = 1
            adjust_factor = 1
        else:
            scale = min_val / 2
            power = np.floor(np.log10(np.abs(min_val)))
            adjust_factor = 1
            shift = -scale
    else:
        scale = max_val - min_val
        power = np.ceil(np.log10((np.abs(min_val) + np.abs(max_val)) / 2))
        adjust_factor = size[1] / scale
        shift = -min_val

    # Determine the multiplier to scale the tick values above 1.0
    multiplier = 10**-power

    # Adjust the title to include the multiplier
    title += f', 10^{int(power)}'

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
    img_array[
        top_left[1] : bottom_right[1] + 1 : pxdelta,
        top_left[0] : bottom_right[0] + 1,
        :,
    ] = grid_color
    img_array[
        top_left[1] : bottom_right[1] + 1,
        top_left[0] : bottom_right[0] + 1 : pxdelta,
        :,
    ] = grid_color

    # Render points
    point_color = point_color

    # Create an array of indices
    i = np.arange(len(adjusted_values))
    x = bottom_right[0] - ((i + 1) * width // len(adjusted_values))
    y = bottom_right[1] - (adjusted_values).astype(int)

    # Create a mask for valid indices
    valid_mask = (
        (top_left[0] <= x)
        & (x <= bottom_right[0])
        & (top_left[1] <= y)
        & (y <= bottom_right[1])
    )
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

    if not scatter and values.shape[0] >= 2:
        # Create pairs of adjacent points
        with _veclinesperf:
            img_array = vectorized_lines_with_thickness(
                y[:-1],
                x[:-1],
                y[1:],
                x[1:],
                img_array,
                clr=line_color,
                thickness=thickness,
            )

    # rendering text
    font_size_small = default_font_size_small
    n = height // (2 * pxdelta)
    tick_color = label_color
    for i in range(n + 1):
        # Scale the tick label by the multiplier
        val = '{:.{}f}'.format((scale / n * i) * multiplier, precision)
        text_size, _ = cv2.getTextSize(val, font, font_size_small, 1)
        text_width, text_height = text_size
        text_x = top_left[0] - text_width  # Adjust position to the left of the grid
        text_y = bottom_right[1] - i * 2 * pxdelta + pxdelta // 2
        cv2.putText(
            img_array, val, (text_x, text_y), font, font_size_small, tick_color, 3
        )

    # Draw title with opacity
    title_color = label_color
    text_size_title = cv2.getTextSize(title, font, font_size, 1)[0]
    text_x_title = top_left[0] + width // 2 - text_size_title[0] // 2
    text_y_title = top_left[1] - text_size_title[1] - pxdelta // 2
    cv2.putText(
        img_array, title, (text_x_title, text_y_title), font, font_size, title_color, 3
    )

    return img_array

@debug_performance('_plotperf')
def plot1_at(
    img_array: np.ndarray,
    values: np.ndarray,
    title: str = 'Measuring',
    offset: Tuple[int, int] = (50, 50),
    size: Tuple[int, int] = (300, 270),
    point_color: Tuple[int, int, int, int] = (0, 0, 255),
    r=2,
    pxdelta: int = 15,
    grid_color: Tuple[int, int, int, int] = (128, 128, 128),
    precision: int = 2,
    default_font_size: float = 0.75,
    default_font_size_small: float = 0.5,
    label_color: Tuple[int, int, int, int] = (0, 0, 255),
    scatter=False,
    thickness=2,
    line_color: Tuple[int, int, int, int] = (0, 0, 255),
    max_len: int = 100,
) -> np.ndarray:
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

    """
    min_val = np.min(values)
    max_val = np.max(values)

    if max_len > 0:
        values = values[-max_len:]

    # Calculate adjustment factor and shift
    if max_val - min_val == 0:
        if min_val == 0:
            scale = 1.0
            shift = -0.5
            power = 1
            adjust_factor = 1
        else:
            scale = min_val / 2
            power = np.floor(np.log10(np.abs(min_val)))
            adjust_factor = 1
            shift = -scale
    else:
        scale = max_val - min_val
        power = np.ceil(np.log10((np.abs(min_val) + np.abs(max_val)) / 2))
        adjust_factor = size[1] / scale
        shift = -min_val

    # Determine the multiplier to scale the tick values above 1.0
    multiplier = 10**-power

    # Adjust the title to include the multiplier
    title += f', 10^{int(power)}'

    # Adjust values
    adjusted_values = (values + shift) * adjust_factor

    # Draw the plot
    height = size[1]
    width = size[0]

    top_left = (offset[0], offset[1])
    bottom_right = (offset[0] + width, offset[1] + height)

    font_size = default_font_size

    # Draw grid and rectangle with opacity
    img_array[
        top_left[1] : bottom_right[1] + 1 : pxdelta,
        top_left[0] : bottom_right[0] + 1,
        :,
    ] = grid_color
    img_array[
        top_left[1] : bottom_right[1] + 1,
        top_left[0] : bottom_right[0] + 1 : pxdelta,
        :,
    ] = grid_color

    # Render points
    point_color = point_color

    # Create an array of indices
    i = np.arange(len(adjusted_values))
    x = bottom_right[0] - ((i + 1) * width // len(adjusted_values))
    y = bottom_right[1] - (adjusted_values).astype(int)

    # Create a mask for valid indices
    valid_mask = (
        (top_left[0] <= x)
        & (x <= bottom_right[0])
        & (top_left[1] <= y)
        & (y <= bottom_right[1])
    )
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

    if not scatter and values.shape[0] >= 2:
        # Create pairs of adjacent points
        with _veclinesperf:
            img_array = vectorized_lines_with_thickness(
                y[:-1],
                x[:-1],
                y[1:],
                x[1:],
                img_array,
                clr=line_color,
                thickness=thickness,
            )

    # rendering text
    font_size_small = default_font_size_small
    n = height // (2 * pxdelta)
    tick_color = label_color[:3]  # Remove alpha channel for vectorized_text
    for i in range(n + 1):
        # Scale the tick label by the multiplier
        val = '{:.{}f}'.format((scale / n * i) * multiplier, precision)
        text_x = top_left[0] - len(val) * 5 * int(font_size_small * 2)  # Approximate text width
        text_y = bottom_right[1] - i * 2 * pxdelta + pxdelta // 2
        img_array = vectorized_text(
            img_array, val, (text_x, text_y), color=tick_color, scale=int(font_size_small * 2)
        )

    # Draw title with opacity
    title_color = label_color[:3]  # Remove alpha channel for vectorized_text
    text_x_title = top_left[0] + width // 2 - len(title) * 5 * int(font_size * 2) // 2  # Approximate text width
    text_y_title = top_left[1] - 15 * int(font_size * 2) - pxdelta // 2  # Approximate text height
    img_array = vectorized_text(
        img_array, title, (text_x_title, text_y_title), color=title_color, scale=int(font_size * 2)
    )

    return img_array

@debug_performance('_plotperf')
def plot1_components(
    values: np.array,
    bounds: Optional[np.ndarray] = None,
    title: str = 'Measuring',
    size: Tuple[int, int] = (300, 300),
    point_color: Tuple[int, int, int, int] = (0, 0, 255, 255),
    r=2,
    nticks: int = 16,
    grid_color: Tuple[int, int, int, int] = (128, 128, 128, 255),
    precision: int = 2,
    default_font_size: float = 0.8,
    default_font_size_small: float = 0.6,
    label_color: Tuple[int, int, int, int] = (0, 0, 255, 255),
    scatter=True,
    thickness=2,
    line_color: Tuple[int, int, int, int] = (0, 0, 255, 255),
    max_len: int = 100,
    *args,
    **kwargs
) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Create a plot and return its components as separate NumPy arrays.

    Args:
        (same as plot1 function)
        bounds: Optional[np.ndarray], custom bounds for the plot (shape: (2, 2) for min/max of x and y)

    Returns:
        Tuple[np.array, np.array, np.array, np.array]: 
            - figure: The main plot content (points and lines)
            - grid: The grid lines
            - labels: The axis labels
            - title_img: The title image
    """
    if max_len > 0:
        values = values[:, -max_len:]
    
    # Estimate margins and grid size
    margin_ver = int(size[1] * 0.15)  # 15% of height for vertical margin
    margin_hor = int(size[0] * 0.15)  # 15% of width for horizontal margin
    grid_topleft = np.array((margin_hor, margin_ver))
    grid_botright = np.array(size) - grid_topleft
    gsize = grid_botright - grid_topleft
    
    # Adjust grid size to be divisible by nticks
    gsize2 = gsize - (gsize % nticks)
    iota = (gsize - gsize2) / 2
    grid_topleft = (grid_topleft + iota).astype(int)
    grid_botright = (grid_botright - iota).astype(int)
    gsize = gsize2
    
    pxdelta = (gsize // nticks).astype(int)

    if bounds is None:
        bounds = np.array([
            [np.min(values[0]), np.max(values[0])],
            [np.min(values[1]), np.max(values[1])]
        ])
    else:
        bounds = np.array(bounds, copy=True)
        # Ensure bounds cover both dimensions
        if bounds.shape != (2, 2):
            raise ValueError("Bounds should have shape (2, 2) for min/max of x and y")
        
        # If any bound is None, calculate it from the values
        for i in range(2):
            if bounds[i, 0] is None:
                bounds[i, 0] = np.min(values[i])
            if bounds[i, 1] is None:
                bounds[i, 1] = np.max(values[i])

    value_range = bounds[:,1] - bounds[:,0]
    scale = gsize[::-1] / value_range
    
    # Correct the broadcasting for adjusted_values
    adjusted_values = (values - bounds[:,0][:, np.newaxis]) * scale[:, np.newaxis]

    # Calculate the order of magnitude for each bound
    magnitude = np.floor(np.log10(np.abs(bounds).astype(float)))
    # Take the maximum magnitude for each dimension
    max_magnitude = np.max(magnitude, axis=0)

    # Update title with the power of 10 for the y-axis (second dimension)
    title += f', 10^{int(max_magnitude[1])}'

    # Create separate image arrays for each component
    figure = np.zeros((*size, 4), np.uint8)
    grid = np.zeros((*size, 4), np.uint8)
    labels = np.zeros((*size, 4), np.uint8)
    title_img = np.zeros((*size, 4), np.uint8)

    # Draw grid
    grid[
        grid_topleft[0] : grid_botright[0] + 1 : pxdelta[0],
        grid_topleft[1] : grid_botright[1] + 1,
        :,
    ] = grid_color
    grid[
        grid_topleft[0] : grid_botright[0] + 1,
        grid_topleft[1] : grid_botright[1] + 1 : pxdelta[1],
        :,
    ] = grid_color

    # Render points
    x = grid_topleft[1] + adjusted_values[0].astype(int)
    y = grid_botright[0] - adjusted_values[1].astype(int)

    valid_mask = (
        (grid_topleft[0] < y) &
        (y < grid_botright[0]) &
        (grid_topleft[1] < x) &
        (x < grid_botright[1])
    )
    valsx = x[valid_mask]
    valsy = y[valid_mask]
    x_offset = np.arange(-r, r + 1)
    y_offset = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(x_offset, y_offset)

    xx = xx.ravel() + valsx[:, None]
    yy = yy.ravel() + valsy[:, None]

    xx = xx.ravel()
    yy = yy.ravel()

    figure[yy, xx] = point_color

    if not scatter and values.shape[1] >= 2:
        with _veclinesperf:
            figure = vectorized_lines_with_thickness(
                y[:-1], x[:-1], y[1:], x[1:],
                figure,
                clr=line_color,
                thickness=thickness,
            )

    # Render y-axis labels
    tick_color = label_color[:3]  # Remove alpha channel for vectorized_text
    for i in range(nticks + 1):
        tick_value = bounds[1, 0] + (value_range[1] * i / nticks)
        val = '{:.{}f}'.format(tick_value, precision)
        l = len(val)
        dx = int(l * 5 * default_font_size_small * 2)
        text_x = grid_topleft[1] - dx  # Approximate text width
        text_y = grid_botright[0] - i * pxdelta[0] - int(5 * default_font_size_small)  # Adjust for text height
        
        labels = vectorized_text(
            labels, val, (text_x, text_y), color=tick_color, font_size=default_font_size_small
        )

    # Render x-axis labels
    for i in range(nticks + 1):
        tick_value = bounds[0, 0] + (value_range[0] * i / nticks)
        val = '{:.{}f}'.format(tick_value, precision)
        l = len(val)
        dy = int(5 * default_font_size_small * 2)
        text_x = grid_topleft[1] + i * pxdelta[1] - int(l * 2.5 * default_font_size_small)  # Center text
        text_y = grid_botright[0] + dy  # Position below x-axis
        
        labels = vectorized_text(
            labels, val, (text_x, text_y), color=tick_color, font_size=default_font_size_small
        )

    # Draw title
    title_color = label_color[:3]  # Remove alpha channel for vectorized_text
    text_x_title = grid_topleft[1] + (grid_botright[1] - grid_topleft[1]) // 2 - len(title) * 5 * int(default_font_size * 2) // 2  # Approximate text width
    text_y_title = grid_topleft[0] - int(default_font_size * 5*2)
    title_img = vectorized_text(
        title_img, title, (text_x_title, text_y_title), color=title_color, font_size=default_font_size
    )

    return figure, grid, labels, title_img

@debug_performance('_plotperf')
def plot(values: np.array,
         grid: dict = None,
         figure: dict = None,
         title: str = 'Plot',
         size: Tuple[int, int] = (300, 300),
         bounds: Optional[np.ndarray] = None,
         max_len: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate plot components as separate RGBA numpy arrays.

    Creates a plot from input values with customizable grid and figure options.
    Returns separate arrays for figure, grid, labels and title that can be blended together.

    Parameters
    ----------
    values : np.array
        2D array of shape (2, N) containing x and y coordinates
    grid : dict, optional
        Grid customization options:

        nticks : int
            Number of grid divisions (default: 10)

        color : tuple
            RGBA color for grid lines (default: (128, 128, 128, 255))

        label_color : tuple
            RGBA color for axis labels (default: (0, 0, 255, 255))

        label_font_size : float
            Font size for axis labels (default: 0.4)

        precision : int
            Decimal precision for axis labels (default: 2)

        title_margin : int
            Margin above title in pixels (default: 30)

        y_tick_offset : int
            Offset for y-axis labels (default: 5)

        x_tick_offset : int
            Offset for x-axis labels (default: 5)

    figure : dict, optional
        Figure customization options:

        scatter : bool
            Whether to draw points (default: False)

        line_color : tuple
            RGBA color for lines (default: (255, 0, 0, 255))

        line_width : int
            Width of lines in pixels (default: 2)

        point_color : tuple
            RGBA color for points (default: (0, 255, 0, 255))

        point_radius : int
            Radius of points in pixels (default: 3)

        marker_style : str
            Point marker style ('circle', 'cross', etc) (default: 'circle')

        line_thickness : int
            Thickness of connecting lines (default: 2)

        line_style : str
            Line style ('solid', 'dashed', etc) (default: 'solid')

    title : str, optional
        Plot title (default: 'Plot')
    size : tuple of int
        (width, height) of plot in pixels (default: (300, 300))
    bounds: Optional[np.ndarray] = None,
        Custom bounds for the plot (shape==(values.shape[-1], 2) for min/max of x and y)
    max_len : int, optional
        Maximum number of points to plot (default: 100)

    Returns
    -------
    tuple of np.ndarray
        (figure_array, grid_array, labels_array, title_array)
        Each array has shape (height, width, 4) with RGBA channels

    Examples
    --------
    >>> x = np.linspace(0, 10, 50)
    >>> y = np.sin(x)
    >>> plot_arrays = plot(np.array([x, y]))
    >>> final_image = blend(*plot_arrays)
    """
    # Default values from plot1_components
    default_grid = {
        'nticks': 16,
        'color': (128, 128, 128, 255),
        'label_color': (0, 0, 255, 255),
        'label_font_size': 0.6,
        'precision': 2,
        'title_margin': 30,
        'y_tick_offset': 5,
        'x_tick_offset': 5
    }

    default_figure = {
        'scatter': True,
        'point_color': (0, 0, 255, 255),
        'point_radius': 2,
        'line_color': (0, 0, 255, 255),
        'line_thickness': 2,
        'marker_style': 'circle',
        'line_style': 'solid'
    }

    # Create new dicts with defaults, updated by user-provided values
    grid_style = default_grid.copy()
    if grid is not None:
        grid_style.update(grid)

    figure_style = default_figure.copy()
    if figure is not None:
        figure_style.update(figure)

    if max_len > 0:
        values = values[:, -max_len:]
    
    # Extract grid options
    nticks = grid_style['nticks']
    grid_color = grid_style['color']
    label_color = grid_style['label_color']
    label_font_size = grid_style['label_font_size']
    precision = grid_style['precision']
    title_margin = grid_style['title_margin']
    y_tick_offset = grid_style['y_tick_offset']
    x_tick_offset = grid_style['x_tick_offset']

    # Extract figure options
    scatter = figure_style['scatter']
    point_color = figure_style['point_color']
    r = figure_style['point_radius']
    line_color = figure_style['line_color']
    thickness = figure_style['line_thickness']
    marker_style = figure_style['marker_style']
    line_style = figure_style['line_style']

    # Estimate margins and grid size
    margin_ver = int(size[1] * 0.15)
    margin_hor = int(size[0] * 0.15)
    grid_topleft = np.array((margin_hor, margin_ver))
    grid_botright = np.array(size) - grid_topleft
    gsize = grid_botright - grid_topleft
    
    # Adjust grid size to be divisible by nticks
    gsize2 = gsize - (gsize % nticks)
    iota = (gsize - gsize2) / 2
    grid_topleft = (grid_topleft + iota).astype(int)
    grid_botright = (grid_botright - iota).astype(int)
    gsize = gsize2
    
    pxdelta = (gsize // nticks).astype(int)
   
    if bounds is None:
        bounds = np.array([
            [np.min(values[0]), np.max(values[0])],
            [np.min(values[1]), np.max(values[1])]
        ])
    else:
        bounds = np.array(bounds, copy=True)
        if bounds.shape != (2, 2):
            raise ValueError("Bounds should have shape (2, 2) for min/max of x and y")
        
        for i in range(2):
            if bounds[i, 0] is None:
                bounds[i, 0] = np.min(values[i])
            if bounds[i, 1] is None:
                bounds[i, 1] = np.max(values[i])

    value_range = bounds[:,1] - bounds[:,0]
    scale = gsize[::-1] / value_range
    
    adjusted_values = (values - bounds[:,0][:, np.newaxis]) * scale[:, np.newaxis]

    # Calculate y magnitude based on the range of y values
    y_range = bounds[1, 1] - bounds[1, 0]
    y_magnitude = int(np.floor(np.log10(y_range))) if y_range != 0 else 0

    title += f', 10^{y_magnitude}'

    figure_img = np.zeros((*size, 4), np.uint8)
    grid_img = np.zeros((*size, 4), np.uint8)
    labels_img = np.zeros((*size, 4), np.uint8)
    title_img = np.zeros((*size, 4), np.uint8)

    # Draw grid
    grid_img[
        grid_topleft[0] : grid_botright[0] + 1 : pxdelta[0],
        grid_topleft[1] : grid_botright[1] + 1,
        :,
    ] = grid_color
    grid_img[
        grid_topleft[0] : grid_botright[0] + 1,
        grid_topleft[1] : grid_botright[1] + 1 : pxdelta[1],
        :,
    ] = grid_color

    # Render points
    x = grid_topleft[1] + adjusted_values[0].astype(int)
    y = grid_botright[0] - adjusted_values[1].astype(int)

    if not scatter and values.shape[1] >= 2:
        # with _veclinesperf:
        figure_img = vectorized_lines_with_thickness(
            y[:-1], x[:-1], y[1:], x[1:],
            figure_img,
            clr=line_color,
            thickness=thickness,
        )
            
    valid_mask = (
        (grid_topleft[0] <= y) & (y < grid_botright[0]) &
        (grid_topleft[1] <= x) & (x < grid_botright[1])
    )
    valsx = x[valid_mask]
    valsy = y[valid_mask]
    
    if marker_style == 'circle':
        x_offset = np.arange(-r, r + 1)
        y_offset = np.arange(-r, r + 1)
        xx, yy = np.meshgrid(x_offset, y_offset)
        mask = xx**2 + yy**2 <= r**2
    elif marker_style == 'square':
        x_offset = np.arange(-r, r + 1)
        y_offset = np.arange(-r, r + 1)
        xx, yy = np.meshgrid(x_offset, y_offset)
        mask = np.ones_like(xx, dtype=bool)
    elif marker_style == 'triangle':
        x_offset = np.arange(-r, r + 1)
        y_offset = np.arange(-r, r + 1)
        xx, yy = np.meshgrid(x_offset, y_offset)
        mask = (yy <= 0) & (xx + yy >= -r) & (-xx + yy >= -r)
    elif marker_style == 'cross':
        x_offset = np.arange(-r, r + 1)
        y_offset = np.arange(-r, r + 1)
        xx, yy = np.meshgrid(x_offset, y_offset)
        mask = (xx == 0) | (yy == 0)
    else:
        # Default to circle if unsupported style is specified
        x_offset = np.arange(-r, r + 1)
        y_offset = np.arange(-r, r + 1)
        xx, yy = np.meshgrid(x_offset, y_offset)
        mask = xx**2 + yy**2 <= r**2

    xx = xx[mask]
    yy = yy[mask]

    xx = xx.reshape(1, -1) + valsx.reshape(-1, 1)
    yy = yy.reshape(1, -1) + valsy.reshape(-1, 1)

    valid_points = (0 <= xx) & (xx < figure_img.shape[1]) & (0 <= yy) & (yy < figure_img.shape[0])
    figure_img[yy[valid_points], xx[valid_points]] = point_color


    # Render y-axis labels
    tick_color = label_color[:3]
    for i in range(nticks + 1):
        tick_value = bounds[1, 0] + (value_range[1] * i / nticks)
        val = '{:.{}f}'.format(tick_value, precision)
        l = len(val)
        dx = int(l * 5 * label_font_size * 2)
        text_x = grid_topleft[1] - dx - y_tick_offset  # Apply y-axis offset
        text_y = grid_botright[0] - i * pxdelta[0] - int(5 * label_font_size)
        
        labels_img = vectorized_text(
            labels_img, val, (text_x, text_y), color=tick_color, font_size=label_font_size, spacing=0.2
        )

    # Render x-axis labels
    for i in range(nticks + 1):
        tick_value = bounds[0, 0] + (value_range[0] * i / nticks)
        val = '{:.{}f}'.format(tick_value, precision)
        l = len(val)
        dy = int(5 * label_font_size * 2)
        text_x = grid_topleft[1] + i * pxdelta[1] - int(l * 2.5 * label_font_size)
        text_y = grid_botright[0] + dy + x_tick_offset  # Apply x-axis offset
        
        labels_img = vectorized_text(
            labels_img, val, (text_x, text_y), color=tick_color, font_size=label_font_size, spacing=0.1 
        )

    # Draw title
    title_color = label_color[:3]
    text_x_title = grid_topleft[1] + (grid_botright[1] - grid_topleft[1]) // 2 - len(title) * 5 * int(label_font_size * 2) // 2
    # Adjust the title's y-position to be closer to the grid
    text_y_title = grid_topleft[0] - int(label_font_size*10)  # Reduced margin

    title_img = vectorized_text(
        title_img, title, (text_x_title, text_y_title), color=title_color, font_size=label_font_size, spacing=0.2
    )

    return figure_img, grid_img, labels_img, title_img

if PIL_available:
    from PIL import Image
    from io import BytesIO
    def blend2PIL(arrays, format='PNG') -> BytesIO:
        """Blend multiple arrays into a PIL image buffer.

        Optimized blending function for Jupyter notebook display that converts
        plot components directly to a PIL image buffer. Requires the Pillow (PIL)
        package to be installed.

        Parameters
        ----------
        arrays : tuple of np.ndarray
            Tuple of RGBA arrays to blend:
            - figure_array: Plot figure components
            - grid_array: Grid lines and background
            - labels_array: Axis labels and ticks
            - title_array: Plot title
            Each array should have shape (height, width, 4) with RGBA channels

        format : str, optional
            Output image format ('PNG', 'JPEG', etc) (default: 'PNG')

        Returns
        -------
        BytesIO
            Buffer containing the blended image in specified format

        Raises
        ------
        ImportError
            If Pillow package is not installed
        ValueError
            If input arrays have different shapes

        Examples
        --------
        >>> plot_arrays = plot(np.array([x, y]))
        >>> buffer = blend2PIL(plot_arrays)
        >>> display(Image(buffer.getvalue()))  # Jupyter display
        """
        figure_pil = Image.fromarray(arrays[0], 'RGBA')
        grid_pil = Image.fromarray(arrays[1], 'RGBA')
        labels_pil = Image.fromarray(arrays[2], 'RGBA')
        title_pil = Image.fromarray(arrays[3], 'RGBA')
        blended_img = Image.alpha_composite(grid_pil, figure_pil)
        blended_img = Image.alpha_composite(blended_img, labels_pil)
        blended_img = Image.alpha_composite(blended_img, title_pil)
        buffer = BytesIO()
        blended_img.save(buffer, format=format)
        return buffer


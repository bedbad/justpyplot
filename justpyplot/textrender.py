import numpy as np
from typing import Tuple

# Define a 5x5 pixel font for uppercase letters, numbers, and some punctuation
FONT = {
    'A': np.array([
        [0,1,1,0,0],
        [1,0,0,1,0],
        [1,1,1,1,0],
        [1,0,0,1,0],
        [1,0,0,1,0]
    ]),
    'B': np.array([
        [1,1,1,0,0],
        [1,0,0,1,0],
        [1,1,1,0,0],
        [1,0,0,1,0],
        [1,1,1,0,0]
    ]),
    'C': np.array([
        [0,1,1,1,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [0,1,1,1,0]
    ]),
    'D': np.array([
        [1,1,1,0,0],
        [1,0,0,1,0],
        [1,0,0,1,0],
        [1,0,0,1,0],
        [1,1,1,0,0]
    ]),
    'E': np.array([
        [1,1,1,1,0],
        [1,0,0,0,0],
        [1,1,1,0,0],
        [1,0,0,0,0],
        [1,1,1,1,0]
    ]),
    'F': np.array([
        [1,1,1,1,0],
        [1,0,0,0,0],
        [1,1,1,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0]
    ]),
    'G': np.array([
        [0,1,1,1,0],
        [1,0,0,0,0],
        [1,0,1,1,0],
        [1,0,0,1,0],
        [0,1,1,1,0]
    ]),
    'H': np.array([
        [1,0,0,1,0],
        [1,0,0,1,0],
        [1,1,1,1,0],
        [1,0,0,1,0],
        [1,0,0,1,0]
    ]),
    'I': np.array([
        [1,1,1,0,0],
        [0,1,0,0,0],
        [0,1,0,0,0],
        [0,1,0,0,0],
        [1,1,1,0,0]
    ]),
    'J': np.array([
        [0,0,1,1,0],
        [0,0,0,1,0],
        [0,0,0,1,0],
        [1,0,0,1,0],
        [0,1,1,0,0]
    ]),
    'K': np.array([
        [1,0,0,1,0],
        [1,0,1,0,0],
        [1,1,0,0,0],
        [1,0,1,0,0],
        [1,0,0,1,0]
    ]),
    'L': np.array([
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,1,1,1,0]
    ]),
    'M': np.array([
        [1,0,0,0,1],
        [1,1,0,1,1],
        [1,0,1,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1]
    ]),
    'N': np.array([
        [1,0,0,0,1],
        [1,1,0,0,1],
        [1,0,1,0,1],
        [1,0,0,1,1],
        [1,0,0,0,1]
    ]),
    'O': np.array([
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0]
    ]),
    'P': np.array([
        [1,1,1,0,0],
        [1,0,0,1,0],
        [1,1,1,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0]
    ]),
    'Q': np.array([
        [0,1,1,0,0],
        [1,0,0,1,0],
        [1,0,0,1,0],
        [1,0,1,0,0],
        [0,1,0,1,0]
    ]),
    'R': np.array([
        [1,1,1,0,0],
        [1,0,0,1,0],
        [1,1,1,0,0],
        [1,0,1,0,0],
        [1,0,0,1,0]
    ]),
    'S': np.array([
        [0,1,1,1,0],
        [1,0,0,0,0],
        [0,1,1,0,0],
        [0,0,0,1,0],
        [1,1,1,0,0]
    ]),
    'T': np.array([
        [1,1,1,1,1],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0]
    ]),
    'U': np.array([
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0]
    ]),
    'V': np.array([
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,0,1,0],
        [0,0,1,0,0]
    ]),
    'W': np.array([
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,1,0,1],
        [1,0,1,0,1],
        [0,1,0,1,0]
    ]),
    'X': np.array([
        [1,0,0,0,1],
        [0,1,0,1,0],
        [0,0,1,0,0],
        [0,1,0,1,0],
        [1,0,0,0,1]
    ]),
    'Y': np.array([
        [1,0,0,0,1],
        [0,1,0,1,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0]
    ]),
    'Z': np.array([
        [1,1,1,1,1],
        [0,0,0,1,0],
        [0,0,1,0,0],
        [0,1,0,0,0],
        [1,1,1,1,1]
    ]),
    '0': np.array([
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0]
    ]),
    '1': np.array([
        [0,0,1,0,0],
        [0,1,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,1,1,1,0]
    ]),
    '2': np.array([
        [0,1,1,1,0],
        [1,0,0,0,1],
        [0,0,1,1,0],
        [0,1,0,0,0],
        [1,1,1,1,1]
    ]),
    '3': np.array([
        [1,1,1,1,0],
        [0,0,0,0,1],
        [0,1,1,1,0],
        [0,0,0,0,1],
        [1,1,1,1,0]
    ]),
    '4': np.array([
        [0,0,1,1,0],
        [0,1,0,1,0],
        [1,0,0,1,0],
        [1,1,1,1,1],
        [0,0,0,1,0]
    ]),
    '5': np.array([
        [1,1,1,1,1],
        [1,0,0,0,0],
        [1,1,1,1,0],
        [0,0,0,0,1],
        [1,1,1,1,0]
    ]),
    '6': np.array([
        [0,1,1,1,0],
        [1,0,0,0,0],
        [1,1,1,1,0],
        [1,0,0,0,1],
        [0,1,1,1,0]
    ]),
    '7': np.array([
        [1,1,1,1,1],
        [0,0,0,0,1],
        [0,0,0,1,0],
        [0,0,1,0,0],
        [0,1,0,0,0]
    ]),
    '8': np.array([
        [0,1,1,1,0],
        [1,0,0,0,1],
        [0,1,1,1,0],
        [1,0,0,0,1],
        [0,1,1,1,0]
    ]),
    '9': np.array([
        [0,1,1,1,0],
        [1,0,0,0,1],
        [0,1,1,1,1],
        [0,0,0,0,1],
        [0,1,1,1,0]
    ]),
    '.': np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,1,0,0]
    ]),
    ',': np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,1,0,0,0]
    ]),
    '!': np.array([
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,1,0,0]
    ]),
    '?': np.array([
        [0,1,1,1,0],
        [1,0,0,0,1],
        [0,0,0,1,0],
        [0,0,0,0,0],
        [0,0,1,0,0]
    ]),
    '-': np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [1,1,1,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ]),
    '+': np.array([
        [0,0,1,0,0],
        [0,0,1,0,0],
        [1,1,1,1,1],
        [0,0,1,0,0],
        [0,0,1,0,0]
    ]),
    '=': np.array([
        [0,0,0,0,0],
        [1,1,1,1,0],
        [0,0,0,0,0],
        [1,1,1,1,0],
        [0,0,0,0,0]
    ]),
    ':': np.array([
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0]
    ]),
    ';': np.array([
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,1,0,0,0]
    ]),
    '(': np.array([
        [0,0,1,0,0],
        [0,1,0,0,0],
        [0,1,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,0]
    ]),
    ')': np.array([
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,0,0,1,0],
        [0,0,0,1,0],
        [0,0,1,0,0]
    ]),
    '[': np.array([
        [0,1,1,0,0],
        [0,1,0,0,0],
        [0,1,0,0,0],
        [0,1,0,0,0],
        [0,1,1,0,0]
    ]),
    ']': np.array([
        [0,0,1,1,0],
        [0,0,0,1,0],
        [0,0,0,1,0],
        [0,0,0,1,0],
        [0,0,1,1,0]
    ]),
    '/': np.array([
        [0,0,0,0,1],
        [0,0,0,1,0],
        [0,0,1,0,0],
        [0,1,0,0,0],
        [1,0,0,0,0]
    ]),
    '\\': np.array([
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,0,0,0,1]
    ]),
    ' ': np.zeros((5, 5))
}

def render_text(text: str, scale: int = 1) -> np.ndarray:
    """Render the entire text as a single numpy array."""
    if not text:
        return np.zeros((5 * scale, 1), dtype=np.uint8)  # Return a minimal array for empty text
    
    char_arrays = [FONT.get(char.upper(), np.zeros((5, 5))) for char in text]
    text_array = np.hstack(char_arrays)
    return np.kron(text_array, np.ones((scale, scale)))

def vectorized_text(
    img_array: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    font_size: float = 0.5,
    spacing: int = 1
) -> np.ndarray:
    """
    Render text onto a NumPy array using optimized vectorized operations.

    Args:
        img_array (np.ndarray): The input image as a NumPy array.
        text (str): The text to render.
        position (Tuple[int, int]): The (x, y) position to place the text.
        color (Tuple[int, int, int]): RGB color of the text.
        font_size (float): Font size, similar to CV2's font scale.
        spacing (int): Spacing between characters.

    Returns:
        np.ndarray: The image array with the text rendered on it.
    """
    x, y = position
    
    # Calculate scale based on font_size
    scale = max(1, int(font_size * 2))
    
    # Render the entire text at once
    text_array = render_text(text, scale)
    
    # Add spacing between characters
    if spacing > 0 and text:
        char_width = 5 * scale
        total_width = text_array.shape[1] + spacing * (len(text) - 1)
        spaced_text_array = np.zeros((text_array.shape[0], total_width), dtype=text_array.dtype)
        char_positions = np.arange(len(text)) * (char_width + spacing)
        spaced_text_array[:, char_positions[:, None] + np.arange(char_width)] = text_array.reshape(text_array.shape[0], -1, char_width)
        text_array = spaced_text_array
    
    # Calculate the region where the text will be placed
    y_start = max(0, y)
    x_start = max(0, x)
    y_end = min(img_array.shape[0], y + text_array.shape[0])
    x_end = min(img_array.shape[1], x + text_array.shape[1])
    
    # Check if the text is completely out of bounds
    if y_end <= y_start or x_end <= x_start:
        return img_array  # Text is completely out of bounds, return original image
    
    # Calculate the visible portion of the text array
    text_y_start = max(0, -y)
    text_x_start = max(0, -x)
    text_y_end = text_y_start + (y_end - y_start)
    text_x_end = text_x_start + (x_end - x_start)
    
    # Crop text_array to the visible portion
    visible_text_array = text_array[text_y_start:text_y_end, text_x_start:text_x_end]
    
    # Get the section of the image we're working with
    img_section = img_array[y_start:y_end, x_start:x_end]
    
    # Create a mask for the text, matching the dimensions of img_section
    mask = np.repeat(visible_text_array[:, :, np.newaxis], img_section.shape[2], axis=2)
    
    # Prepare the color array
    color_array = np.array(color)
    if img_section.shape[2] == 4:  # If the image has an alpha channel
        color_array = np.append(color_array, 255)  # Add full opacity
    
    # Create a color overlay
    color_overlay = np.tile(color_array, (mask.shape[0], mask.shape[1], 1))
    
    # Blend the text with the image
    blended = img_section * (1 - mask) + color_overlay * mask
    
    # Assign the blended result back to the image array
    img_array[y_start:y_end, x_start:x_end] = blended.astype(img_array.dtype)
    
    return img_array

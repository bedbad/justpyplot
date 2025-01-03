import numpy as np
import cv2
from justpyplot.textrender import vectorized_text

def test_text_rendering():
    # Create a blank image
    img_array = np.zeros((300, 600, 3), dtype=np.uint8)

    # Test 1: Basic text rendering
    img_array = vectorized_text(img_array, "JUST TEXT", (50, 50), color=(255, 255, 255), scale=2)

    # Test 2: Colored text
    img_array = vectorized_text(img_array, "RED TEXT", (50, 100), color=(0, 0, 255), scale=2)

    # Test 3: Smaller scale
    img_array = vectorized_text(img_array, "Small Text", (50, 150), color=(0, 255, 0), scale=1)

    # Test 4: Numbers and punctuation
    img_array = vectorized_text(img_array, "123!@#$%", (50, 200), color=(255, 255, 0), scale=2)

    # Test 5: Long text with wrapping
    long_text = "This is a long text that will wrap to the next line"
    words = long_text.split()
    y_position = 250
    for i in range(0, len(words), 3):
        line = " ".join(words[i:i+3])
        img_array = vectorized_text(img_array, line, (50, y_position), color=(255, 0, 255), scale=1)
        y_position += 30

    # Display the result using OpenCV
    cv2.imshow('Text Rendering Test', img_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_text_rendering()
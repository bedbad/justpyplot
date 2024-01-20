![Plot in Numpy Fast](https://giphy.com/embed/Golhlo06fa2Q5EM0o5.gif)
# PLOT IN NUMPY FAST

Justpyplot plots given `values` to a given NumPy ndarray, adapting
the plot scale and size to fit the input data.
Plots fast - the policy is no single loop in the code,
even if when you connect points with line segments etc,
It is measured at 20-100x faster matplotlib.
Useful for overlaying real-time plots on images and video frames, or
in jupyter.
## MOTIVATION
Doing dances around matplotlib by getting figures not auto displayed, converting figure bytearray to numpy(which is different by
backend, i.e OS = surprising behavior on different platforms)
to suffering multimillisecond performance hit just didn't do it for me.
This repo is to see if there is enough need/pain out there to make that right.

## INSTALATION

You can copy plot files respecting the license, or for use in Jupyter notebooks more convienent to:
```bash    
pip install justpyplot
```
## BASIC USAGE

You can explore the documentation for the justplot function currently, options probably support any 
flexing or adjusting the plot as you need such as colors, scatter, connected with line thickness, labels, etc

Basically you just create two matching arrays for the dependency you plot and pass it to the justplot with where your plot to be sticked (numpy array, mask in memory, etc) and it does it. Adjust your function .
```python
import numpy as np 
import cv2

import plot as ap

t0 = time.perf_counter()

xs, ys = [], []
c = 0
while(c != 27):
    xt = time.perf_counter() - t0
    yx = np.sin(xt)
    xs.append(xt)
    ys.append(yx)
    
    frame = np.full((500,470,3), (255,255,255), dtype=np.uint8)
    
    vals = np.array(ys)
    with _plottimer:
        drawn = ap.just_plot(frame, vals,title="sin() from Clock")
    
    cv2.imshow('frame', drawn)
    c = cv2.waitKey(1)
```

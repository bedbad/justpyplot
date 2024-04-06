
<h1 align="center"># Plot in Numpy, Debug Visually Real-Time</h1>

<p align="center">
  <img src="resources/demo.gif">
</p>

This is concept of the real-time, vectorized plotting library
which gets your plot as a numpy array controlling every basic part
of it you may need - including opacity; 

It's useful if you want to take control of your plot rendering/recording in your own hands for grab your plots directly for __overlaying__, GIFing
visual debugging, tweaking, streaming, etc; It allows direct control and access img masks of 4 elemental properties of plot:  __figures__, __grid__, __axis__ and __labels__

It measured at 20-100x performance of matplotlib.
```
timer "full justpyplot + rendering": avg 382 µs ± 135 µs, max 962 µs
```

## MOTIVATION
Python Plotting status quo: matplotlib and other major libs that copy it
need 3-6 step [hoopla](https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array/77853862#77853862) to get as simple numpy arrays __figures__, __grid__, __axis__ and __labels__ of your plot. 

The status quo basically requires you to _screenshot_ in memory and buffer byter data back to pixel/img tensor. That makes you suffer multimillisecond performance hit _every plot frame_

Obviously it's very unpleasant to write such hoopla if you want to frab your plot frames. \
Having plot frames you can do tweaking/publishing/rendering/GIFing etc.. yourself and, _importantly for computer vision_, it's too much a performance looser for real-time aspect of what you plot, such as testing and debugging, physical/manual manipulation of CV/camera/mechanical setup.

Also: it is a hermetic python plotlib rewrite  - it doesn't suffer from other matplotlib vices.\
Also: this code is an instance of plotting/rendering done in vectorized way which is important in ML/AI industry - the plot workflow can be directly embedded in visual neural networks

If there's enough demand here for real-time plotting, visual debugging for camera/computer vision applications and quick image graphs, I will  a solid level of this library, with more competitive basic feature set to that of matplotlib.

## INSTALATION

You can copy justplot file code respecting the license;
For use in Jupyter notebooks its convienent to:
```bash    
pip install justpyplot
```
## BASIC USAGE

![Basic Usage](resources/sinus.gif)

You can explore the documentation for the justplot function currently, options probably support any  flexing or adjusting the plot as you need such as colors, scatter, connected with line thickness, labels, etc

You just pass one or two arrays signifying dependency and pass it to the justplot.
You can plot on exiting image frame at specified offset by plot1_at, plot2_at function or directly generate plot mask, you can completely control opacity of every element as well as every graphical or textual parameter of the plot.

```python
import numpy as np 
import cv2
import time
from justpyplot import justpyplot as jplt

xs, ys = [], []
while(cv2.waitKey(1) != 27):
    xt = time.perf_counter() - t0
    yx = np.sin(xt)
    xs.append(xt)
    ys.append(yx)
    
    frame = np.full((500,470,3), (255,255,255), dtype=np.uint8)
    
    vals = np.array(ys)

    plotted_in_array = jplt.plot1(frame, vals,title="sin() from Clock")
    
    cv2.imshow('np array plot', plotted_in_array)
```

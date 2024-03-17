
<h1 align="center"># Plot in Numpy, Debug Visually Real-Time</h1>

<p align="center">
  <img src="resources/demo.gif">
</p>

This is concept of the real-time, vectorized plotting library
which gets your plot as a numpy array controlling every basic part
of it you may need - including opacity; 

It's useful if you want to take control of your plot rendering/recording in your own hands for grab your plots directly for __overlaying__, GIFing
visual debugging, tweaking, streaming, etc; It allows direct control and access img masks of 4 elemental properties of plot:  __figures__, __grid__, __axis__ and __labels__

Also it is measured at 20-100x faster then matplotlib.
```
timer "full justpyplot + rendering": avg 382 µs ± 135 µs, max 962 µs
```

## MOTIVATION
Python plotting Status Quo, matplotlib and other major libs that copy it require 
3-6 step ![hoopla](https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array/77853862#77853862) to get  the numpy array of __figures__, __grid__, __axis__ and __labels__ from your data you'd like to plot. 

The status quo basically requires you to screenshot and convert img data back to pixel tensor makes you suffer multimillisecond performance hit.

Obviously it's very unpleasant to write such hoopla if you need it to take care of your python plots tweaking/publishing/rendering/gifing etc.. yourself and it's not fit for real-time aspect of what you plot, such as testing and debugging, physical/manual manipulation of CV/camera/mechanical setup.

Because it's hermetic python plotlib rewrite it doesn't suffer from other matplotlib vices.
Also!
This code is a quick example for people who want to see how to do things in vectorized way.

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

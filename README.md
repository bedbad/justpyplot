
<h1 align="center"># Why not just get your plots in numpy?! </h1>
Why not just have a simple plot1(values,size,title, scatter=True, pt_color, ...) -> np.ndarray 

function API which gives your plot (__figures__, __grid__, __axis__,__labels__ if needed) in any way you tell it to?

And then you take that and overlay, mask, stretch, transform and render at wherever in frame array/canvas/memory you need?

Here I took this function concept and stuck it upon running mug neural network in 3 lines of code to see how it performs in reality in details (try in examples/demo.py):
<p align="center">
  <img src="resources/demo.gif">
</p>

```
                jplt.plot1_at(image, ang_zs,
                            title='Angle from Z axis', offset=(50,50), size=(270, 300),
                            point_color=(255,0,0),line_color=(255,0,0), label_color=(255,0,0), grid_color=(126,126,126))
                jplt.plot1_at(image, ang_ys,
                              title='Angle from Y axis', offset=(400,50), size=(270, 300),
                              point_color=(0,255,0), line_color=(0,255,0),label_color=(0,255,0), grid_color=(126,126,126),
                              scatter=False)
                jplt.plot1_at(image,ang_xs,
                              title='Angle from X axis', offset=(750,50), size=(270, 300),
                              point_color=(0,0,255), line_color=(0,0,255),label_color=(0,0,255), grid_color=(126,126,126),
                              scatter=False)
```

It measured at 20-100x performance of matplotlib.
```
timer "full justpyplot + rendering": avg 382 µs ± 135 µs, max 962 µs
```

Because I did the plotting in native, vectorized way which is an interesting problems topic in itself and
This turned out to be a concept of the real-time, vectorized plotting library
which allows you to control controlling every basic part of your graph/plot parts you may need - including the opacity; 

It showed useful in robotics and in general if you want to take control of your plot rendering/recording in your own hands for grab your plots directly for __overlaying__, GIFing
visual debugging, tweaking, streaming, etc; It allows direct control and access img masks of 4 elemental properties of plot:  __figures__, __grid__, __axis__ and __labels__

## WHY USE IT
Python Plotting status quo: matplotlib and other major libs that copy it
need 3 to 6 steps [hoopla](https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array/77853862#77853862) to get your plot data in easy format (as simple numpy arrays) such as __figures__, __grid__, __axis__ and __labels__ of your plot. 

The status quo basically requires you to _screenshot_ in memory and buffer byter data back to pixel/img tensor. That makes you suffer multimillisecond performance hit _every plot frame_

Obviously it's very unpleasant to write such hoopla if you want to grab your plot datas frame by frame. \
Having plot frames you can do tweaking/publishing/rendering/GIFing etc.. yourself and, _importantly for computer vision_, it's too much a performance looser for real-time aspect of what you plot, such as physical/robotic testing and debugging, physical/manual manipulation of CV/camera/mechanical setup.

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

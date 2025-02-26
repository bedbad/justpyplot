import numpy as np
import cv2
import time
from justpyplot import justpyplot as jplt

t0 = time.perf_counter()
xs, ys = [], []
while cv2.waitKey(1) != 27:
    xt = time.perf_counter() - t0
    yx = np.sin(xt)
    xs.append(xt)
    ys.append(yx)

    vals = np.array(ys)
    plotted_array = jplt.plot1(vals, title='sin() from Clock')

    cv2.imshow('np array plot', plotted_array)

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import numpy as np 
import time
import cv2

import perf_timer
import plot as ap


_plottimer = perf_timer.PerfTimer('adaptive plot')

t0 = time.perf_counter()

xs, ys = [], []
c = 0
while(c != 27):
    time.sleep(0.1)
    xt = time.perf_counter() - t0
    yx = np.sin(xt)
    xs.append(xt)
    ys.append(yx)
    frame = np.full((500,470,3), (255,255,255), dtype=np.uint8)
    vals = np.array(ys)
    with _plottimer:
        drawn = ap.just_plot(frame, vals,title="sin() from Clock", offset=(100,100), opacity=1., max_len=200)

    cv2.imshow('frame', drawn)
    c = cv2.waitKey(1)
    if(c == 32):
        c = cv2.waitKey(0)

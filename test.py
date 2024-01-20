# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import numpy as np 
import time
import cv2

import perf_timer
import justpyplot as jplt


_plottimer = perf_timer.PerfTimer('full justpyplot + rendering')

t0 = time.perf_counter()

xs, ys = [], []
c = 0
while(c != 27):
    xt = time.perf_counter() - t0
    yx = np.sin(xt)
    xs.append(xt)
    ys.append(yx)
    frame = np.full((400,400,3), (255,255,255), dtype=np.uint8)
    vals = np.array(ys)
    with _plottimer:
        drawn = jplt.just_plot(frame, vals,title="sin() from Clock", offset=(50,50), scatter=True, max_len=100)

    cv2.imshow('frame', drawn)
    c = cv2.waitKey(1)
    if(c == 32):
        c = cv2.waitKey(0)

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import numpy as np 
import time
import cv2

import perf_timer 
import adaptiveplot as ap

def plot_on_frame(frame:np.ndarray, plot:np.ndarray, offset):
    assert np.all(plot.shape[0:2]+offset<frame.shape[0:2])
    a = np.zeros_like(frame)

_plottimer = perf_timer.PerfTimer('adaptive plot')

t0 = time.perf_counter()

xs, ys = [], []
c = 0
while(c != 27):
    xt = time.perf_counter() - t0
    yx = np.sin(xt)
    xs.append(xt)
    ys.append(yx)
    frame = np.full((800,600,3), (64,128,64), dtype=np.uint8)
    vals = np.array(ys)
    with _plottimer:
        drawn = ap.draw_adaptive_plot(frame, vals, opacity=1., max_len=100)
    # plt.plot(xs, ys)
    # fig = plt.gcf()
    # frame = np.array(fig.canvas.get_renderer().buffer_rgba())
    # print(frame[frame[:,:,...]!=np.array([255,255,255,0])])
    cv2.imshow('frame', drawn)
    c = cv2.waitKey(1)
    # plt.close(fig)


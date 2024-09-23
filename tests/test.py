# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

import perf_timer
from justpyplot import justpyplot as jplt

_plottimer = perf_timer.PerfTimer('full justpyplot + rendering')

t0 = time.perf_counter()

xs, ys = [], []
c = 0
while c != 27:
    xt = time.perf_counter() - t0
    yx = np.sin(xt)
    xs.append(xt)
    ys.append(yx)
    frame = np.full((400, 400, 3), (255, 255, 255), dtype=np.uint8)
    vals = np.array([xs, ys])
    with _plottimer:
        final = jplt.plot2_at(
            img_array=frame, values=vals, offset=(50, 50), title='sin() from Clock'
        )
    # final = jplt.blend_at(frame, plot, offset=(50,50))
    # drawn = jplt.plot1_at(frame, vals,title="sin() from Clock", offset=(50,50), scatter=True, max_len=100)

    cv2.imshow('frame', final)
    c = cv2.waitKey(1)
    if c == 32:
        c = cv2.waitKey(0)

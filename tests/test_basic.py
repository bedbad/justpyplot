# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import numpy as np 
import time
import cv2

from justpyplot import justpyplot as jplt

t0 = time.perf_counter()

xs, ys = [], []
c = 0
while(c != 27):
    xt = time.perf_counter() - t0
    yx = np.sin(xt)
    xs.append(xt)
    ys.append(yx)
    frame = np.full((400,400,3), (255,255,255), dtype=np.uint8)
    vals = np.array([xs,ys])
    final = jplt.plot2_at(img_array=frame,values=vals,offset=(50,50),title="sin() from Clock")

    cv2.imshow('frame', final)
    c = cv2.waitKey(1)

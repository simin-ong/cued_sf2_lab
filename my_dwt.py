%matplotlib nbagg
import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
import numpy as np
from typing import Tuple
from cued_sf2_lab.dwt import dwt, idwt
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import quantise

def multi_dwt(X, n):
    m=X.shape[0]
    Y=dwt(X)
    for i in range(n-1):
        m = m//2
        Y[:m,:m] = dwt(Y[:m,:m])
    return Y

def multi_idwt(Y, n):
    m = int(Y.shape[0]/(2**(n-1)))
    Z = Y.copy()
    Z[:m,:m] = idwt(Y[:m,:m])
    for j in range(n-1):
        m = m*2
        Z[:m,:m] = idwt(Z[:m,:m])
    return Z

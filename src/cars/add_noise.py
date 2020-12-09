import numpy as np


def add_noise(img):

    row,col=28,28
    img=img.astype(np.float32)

    mean=0
    var=10
    sigma=var**.5
    noise=np.random.normal(-5.9,5.9,img.shape)
    noise=noise.reshape(row,col)
    img=img+noise
    return img

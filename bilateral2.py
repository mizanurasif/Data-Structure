
import numpy as np
import cv2 as cv
import math
def spatial_kernel(ksize,sigma):
    kernel = np.zeros((ksize,ksize),dtype='float32')
    k = ksize//2
    cnst=(sigma*sigma*2*math.pi)
    
    for i in range(-k,k+1):
        for j in range(-k,k+1):
            kernel[i+k,j+k]= (math.exp(-(i*i+j*j)/(2*sigma*sigma)))/cnst
    return kernel
def range_kernel(img,x,y,ksize,sigma):
    kernel = np.zeros((ksize,ksize),dtype='float32')
    k = ksize//2
    cnst=(sigma*sigma*2*math.pi)
    p = img[x,y]
    for i in range(-k,k+1):
        for j in range(-k,k+1):
            q= img[x-i+k,y-j+k]      
            pr = pow((p-q),2)
            kernel[i+k,j+k]= (math.exp(-pr/(2*sigma*sigma)))
    return kernel
            
def multiply(sp,rp,ksize):
    kernel = np.zeros((ksize,ksize),dtype='float32')
    for i in  range(ksize):
        for j in range(ksize):
            kernel[i,j]=sp[i,j]*rp[i,j]
    s=kernel.sum()
    kernel= kernel/s
    return kernel
def bilateral(img,ksize,sigma):

    h=img.shape[0]
    w=img.shape[1]
    k=ksize//2
    res=np.zeros((h,w))
    bordered_img=cv.copyMakeBorder(img, k,k,k,k,cv.BORDER_REPLICATE)
    sp = spatial_kernel(ksize, sigma)

    for x in range(h):
        for y in range(w):
            rp = range_kernel(bordered_img, x, y, ksize, sigma)
            kernel =multiply(sp, rp, ksize)
            
            for i in range(-k,k+1):
                for j in range(-k,k+1):                      
                        res[x,y]+=kernel[i+k,j+k]*bordered_img[x-i+k,y-j+k]
                               
    cv.normalize(res,res,0,255,cv.NORM_MINMAX)
    res=np.round(res).astype(np.uint8)
    cv.imshow('output image',res)
    
img = cv.imread('lena.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY )
cv.imshow('input image',img)
bilateral(img, 5, 5)
blur=cv.bilateralFilter(img,3,5,5)
cv.imshow('blurred using opencv',blur)

    
cv.waitKey(0)
cv.destroyAllWindows()



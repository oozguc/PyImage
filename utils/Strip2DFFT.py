

from scipy.signal import blackman
from scipy.fftpack import fft, ifft, fftshift
from scipy.fftpack import fftfreq
import numpy as np


def CrossCorrelationStrip(imageA, imageB):
    
    PointsSample = imageA.shape[1] 
    stripA = imageA[:,0]
    for i in range(imageA.shape[1]):
        
        stripB = imageB[:,i]
        stripCross = CrossCorrelation(stripA, stripB)
        PointsSample += stripCross
    PointsSample = PointsSample/max(PointsSample)
    return PointsSample  

def FFTStrip(imageA):
    ffttotal = np.empty(imageA.shape)
    PointsSample = imageA.shape[1] 
    for i in range(imageA.shape[0]):
        stripA = imageA[i,:]
        
        fftstrip = fftshift(fft(stripA))
        ffttotal[i,:] = np.abs(fftstrip)
    return ffttotal 

#FFT along a strip
def doFilterFFT(image,Time_unit, filter):
   addedfft = 0 
   PointsSample = image.shape[1] 
   for i in range(image.shape[0]):
      if filter == True:   
       w = blackman(PointsSample)
      if filter == False:
       w = 1
      strip = image[i,:]
      fftresult = fft(w * strip)
      addedfft += np.abs(fftresult)  
   #addedfft/=image.shape[0]
   
   
   xf = fftfreq(PointsSample, Time_unit)
   
   
   return addedfft[1:int(PointsSample//2)], xf[1:int(PointsSample//2)]


def do2DFFT(image, Space_unit, Time_unit, filter):
    fftresult = fft(image)
    PointsT = image.shape[1]
    PointsY = image.shape[0]
    Tomega = fftfreq(PointsT, Time_unit)
    Spaceomega = fftfreq(PointsY, Space_unit)
    
    return fftresult

def do2DInverseFFT(image, Space_unit, Time_unit, filter):
    fftresult = ifft(image)
    PointsT = image.shape[1]
    PointsY = image.shape[0]
    Tomega = fftfreq(PointsT, Time_unit)
    Spaceomega = fftfreq(PointsY, Space_unit)
    
    return fftresult
def CrossCorrelation(imageA, imageB):
    crosscorrelation = imageA
    fftresultA = fftshift(fft(imageA))
    fftresultB = fftshift(fft(imageB))
    multifft = fftresultA * np.conj(fftresultB)
    crosscorrelation = fftshift(ifft(multifft))
    return np.abs(crosscorrelation) 
import numpy as np
import imageio
import math
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.measure import LineModelND, ransac

from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from Normalize import save_tiff_imagej_compatible
from skimage.segmentation import find_boundaries,find_boundaries, relabel_sequential
from skimage.morphology import remove_small_objects, binary_erosion
from skimage.filters import threshold_otsu, threshold_mean
from skimage.exposure import rescale_intensity
import os
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def show_ransac_points_line(points,  min_samples=2, residual_threshold=0.1, max_trials=1000):
   
    # fit line using all data
 model = LineModelND()
 model.estimate(points)
 
 fig, ax = plt.subplots()   

 # robustly fit line only using inlier data with RANSAC algorithm
 model_robust, inliers = ransac(points, LineModelND, min_samples=min_samples,
                               residual_threshold=residual_threshold, max_trials=max_trials)
 slope , intercept = model_robust.params
 
 outliers = inliers == False
 # generate coordinates of estimated models
 line_x = np.arange(0, 100)
 line_y = model.predict_y(line_x)
 line_y_robust = model_robust.predict_y(line_x)
 
 #print('Model Fit' , 'yVal = ' , line_y_robust)
 #print('Model Fit', 'xVal = ' , line_x)
 ax.plot(points[:, 0], points[:, 1], '.b', alpha=0.6,
        label='Inlier data')
 
 ax.plot(line_x, line_y, '-r', label='Normal line model')
 ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
 ax.legend(loc='lower left')
    
 print('Slope = ', (line_y_robust[99] - line_y_robust[0])/ (100) ) 
 print('Normal Slope = ', (line_y[99] - line_y_robust[0])/ (100) ) 
 plt.show()
 
    
    

def show_ransac_line(img, Xcalibration, Time_unit, maxlines, min_samples=2, residual_threshold=0.1, max_trials=1000):
    points = np.array(np.nonzero(img)).T

    f, ax = plt.subplots(figsize = (10, 10))

    points = points[:, ::-1]

    for i in range(maxlines):
  
     
    # robustly fit line only using inlier data with RANSAC algorithm
     model_robust, inliers = ransac(points, LineModelND,  min_samples=min_samples,
                               residual_threshold=residual_threshold, max_trials=max_trials)
     slope , intercept = model_robust.params
 
     points = points[~inliers]   

     print("Estimated Wave Velocity by Ransac : " ,np.abs(slope[0])* (Xcalibration / Time_unit)) 
     x0 = np.arange(img.shape[1])   
 
     y0 =  model_robust.predict_y(x0)
 
     plt.plot(x0, model_robust.predict_y(x0), '-r')
 
    ax.imshow(img)
    

def watershed_image(image, size, targetdir, Label, Filename, Xcalibration,Time_unit,low_slope_threshold,high_slope_threshold,intensity_threshold, SupressView = True):
 distance = ndi.distance_transform_edt(image)
 local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=image)
 markers = ndi.label(local_maxi)[0]
 labels = watershed(-distance, markers, mask=image)



    
    
 nonormimg = remove_small_objects(labels, min_size=size, connectivity=4, in_place=False)
 nonormimg, forward_map, inverse_map = relabel_sequential(nonormimg)    
 labels = nonormimg
 Velocity = []

 # loop over the unique labels returned by the Watershed
 # algorithm
 for label in np.unique(labels):
      
      if label== 0:
            continue
     
      mask = np.zeros(image.shape, dtype="uint8")
      mask[labels == label] = 1
        
      h, theta, d = hough_line(mask)  
      velocity = show_hough_linetransform(mask, h, theta, d, Xcalibration, 
                               Time_unit,low_slope_threshold,high_slope_threshold,intensity_threshold, targetdir, Filename[0])

      Velocity.append(velocity)
 return Velocity    

    
def show_hough_linetransform(img, accumulator, thetas, rhos, Xcalibration, Tcalibration, low_slope_threshold,high_slope_threshold,intensity_threshold, save_path=None, File = None, SupressView = True):
    import matplotlib.pyplot as plt

    #fig, ax = plt.subplots(1, 2, figsize=(10, 10))

   

    #ax[0].imshow(
        #accumulator, cmap=cm.gray,
        #extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    
    #ax[0].set_title('Hough transform')
    #ax[0].set_xlabel('Angles (degrees)')
    #ax[0].set_ylabel('Distance (pixels)')
    #ax[0].axis('image')
    #ax[1].imshow(img, cmap=cm.gray)
    
    bestpeak = 0
    bestslope = 0
    besty0 = 0
    besty1 = 0
    Est_vel = []
    for _, angle, dist in zip(*hough_line_peaks(accumulator, thetas, rhos, threshold = intensity_threshold* np.max(accumulator))):
     y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
     y1 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
    
     pixelslope =   -( np.cos(angle) / np.sin(angle) )
     pixelintercept =  dist / np.sin(angle)  
     slope =  -( np.cos(angle) / np.sin(angle) )* (Xcalibration / Tcalibration)
     
    #Draw high slopes
     peak = 0;
     for index, pixel in np.ndenumerate(img):
            x, y = index
            vals = img[x,y]
            if np.abs(y - pixelslope * x - pixelintercept) <= 5:
                peak+=vals
                if peak >= bestpeak:
                    bestpeak = peak
                    bestslope = slope
                    besty0 = y0
                    besty1 = y1
   
    
    
    #ax[1].plot((0, img.shape[1]), (besty0, besty1), '-r')
    
    #ax[1].set_xlim((0, img.shape[1]))
    #ax[1].set_ylim((img.shape[0], 0))
    #ax[1].set_axis_off()
    #ax[1].set_title('Detected lines')

    # plt.axis('off')
    if save_path is not None and File is not None:
       plt.savefig(save_path + 'HoughPlot' + File + '.png')
    if save_path is not None and File is None:
        plt.savefig(save_path + 'HoughPlot' + '.png')
  
    
       #plt.show()

    return np.abs(bestslope)    
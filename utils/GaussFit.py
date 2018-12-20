import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
import os
import math
from copy import deepcopy


import scipy

import pylab
def StripFit(image, Time_unit, Xcalibration):
    
    Fwhm = []
    Time = []
    for i in range(image.shape[1]):
        X = []
        I = []
        strip = image[2:image.shape[0]-2,i]
        
        for j in range(strip.shape[0]):
           X.append(j)
           I.append(strip[j]) 
           
        
        GaussFit = Linescan(X,I)
        GaussFit.extract_ls_parameters()
        fwhm = GaussFit.fwhm
        if fwhm > 0:
            Fwhm.append(fwhm* Xcalibration)
            Time.append(i* Time_unit)
    print('Mean Thickness (Before outlier removal) = ', str('%.3f'%(sum(Fwhm)/len(Fwhm))), 'um')    
    return Fwhm, Time    
        
        
def gauss_func(p, x):
    """Definition of gaussian function used to fit linescan peaks.
    p = [a, sigma, mu, c].
    """
    a, sigma, mu, c = p #unpacks p (for readability)
    g = a / (sigma * math.sqrt(2 * math.pi)) * scipy.exp(-(x - mu)**2 / (2 * sigma**2)) + c
    return g


def convolved(p,x):
    """Defines convolved linescan. Args: x: float or list/iterable of floats,
    the position for which convolved intensity is calculated; p: list/iterable
    of floats, linecan parameters (p=[i_in, i_c, i_out, h, x_c, sigma]).
    Returns: i: float, intensity at x.
    """
    i_in, i_c, i_out, h, x_c, sigma = p #unpacks p (for readability)

    i = (i_in + (i_c - i_in) * stats.norm.cdf((x - x_c) + h / 2., 0., sigma) +
         (i_out - i_c) * stats.norm.cdf((x - x_c) - h / 2., 0., sigma))

    return i

def unconvolved(p,x):
    """Defines unconvolved linescan. Args: x: float or list/iterable of floats,
    the position for which intensity is calculated; p: list/iterable of floats,
    linecan parameters (p=[i_in, i_c, i_out, h, x_c]). Returns: i: float,
    intensity at x.
    """

    i_in, i_c, i_out, h, x_c = p #unpacks p (for readability)

    i = np.zeros(len(x))

    for j in range(len(x)):
        if x[j] < x_c - h / 2.:
            i[j] = i_in
        if x[j] >=  x_c - h / 2. and x[j] <  x_c + h / 2.:
            i[j] = i_c
        if x[j] >= x_c + h / 2.:
            i[j] = i_out

    return i
class Linescan():
    """Linescan object with methods to extract important parameters
    from linescans.
    """

    def __init__(self,x,i):
        """Initializes linescan.
        Args:
            x (list of numbers): the position values
            i (list of numbers): the intensity values
        """
        #populate linescan position/intensity
        self.x = np.asarray(x) #position list as NumPy array of floats
        self.i = np.asarray(i) #intensity list as NumPy array of floats

        #detminere a few easy parameters from position/intensity
        self.H = self.x[-1] - self.x[0]
        self.i_tot = np.trapz(self.i,self.x)

        #populate other attributes
        self.dist_to_x_in_out = 1. #specifies how far away x_in is from the peak (in um)
        self.gauss_params = None #parameter list from Gaussian fit to find peak
        self.x_peak = None #linescan peak position
        self.i_peak = None #linescan peak intensity
        self.i_in = None #intracellular intensity
        self.i_out = None #extracellular intensity
        self.max_idx = None #index of point near linescan center with highest intensity
        self.x_fit = None #position list used for peak fitting
        self.i_fit = None #intensity list used for peak fitting
        self.i_in_x_list = None #position list used to determine self.i_in
        self.i_in_i_list = None #intensity list used to determine self.i_in
        self.i_out_x_list = None #position list used to determine self.i_out
        self.i_out_i_list = None #intensity list used to determine self.i_out
        self.x_in_upper_index = None #the index at the upper end of the region where x_in is calculated
        self.x_out_lower_index = None #the index at the lower end of the region where x_out is calculated
        self.fwhm = None #full width at half-max

        #initializes linescans and determines linescan parameters
        self.extract_ls_parameters()

    def convert_px_to_um(self):
        """Multiplies list of coordinates by pixel_size."""

        self.x = np.array([a * self.px_size for a in self.x])

    def extract_ls_parameters(self):
        """Extracts intensity and position information from linescan"""

        self.get_peak()
        self.get_i_in_out()
        self.get_fwhm()
       
    def get_peak(self):
        """Finds the peak position and intensity of a linescan by fitting
        a Gaussian near the peak.
        """

        #restricts fitting to near the center of the linescan
        self.max_idx = np.argmax(self.i[int(len(self.i)/2-6):int(len(self.i)/2+20)]) + int(len(self.i)/2-6)
        self.x_fit = self.x[self.max_idx-2:self.max_idx+3]
        self.i_fit = self.i[self.max_idx-2:self.max_idx+3]

        #picks reasonable starting values for fit
        self.i_in_guess = np.mean(self.i[:self.max_idx-14])
        a = (self.i[self.max_idx] - self.i_in_guess) / 2.4
        sigma = 0.170
        mu = self.x[self.max_idx]
        b = self.i_in_guess

        #perform fit with starting values
        p0 = [a, sigma, mu, b]
        p1, success  = optimize.leastsq(self.residuals_gauss,p0,
                                        args=(self.x_fit, self.i_fit),
                                        maxfev = 1000000)
        self.gauss_params = p1
        self.x_peak = p1[2]
        self.i_peak = gauss_func(p1, self.x_peak)

    def get_i_in_out(self):
        """Gets values for intracellular intensity (self.i_in) and
        extracellular intensity (self.i_out). The left of the linescan
        (nearer zero) is always assumed to be the intracellular side.
        Note: the i_in and i_out values are calculated to be the average value
        of the ten points out from the distance between the peak and position x away
        from the peak, where x is given by self.dist_to_x_in_out (defined in __init__).
        """

        x_in_upper = self.x_peak - self.dist_to_x_in_out
        x_in_upper_index = np.argmin(abs(self.x - x_in_upper))
        self.x_in_upper_index = x_in_upper_index #for use in finding total intensity for density calculation
        self.i_in_x_list = self.x[x_in_upper_index-10:x_in_upper_index]
        self.i_in_i_list = self.i[x_in_upper_index-10:x_in_upper_index]
        self.i_in = np.mean(self.i_in_i_list)

        x_out_lower = self.x_peak + self.dist_to_x_in_out
        x_out_lower_index = np.argmin(abs(self.x - x_out_lower))
        self.x_out_lower_index = x_out_lower_index #for use in finding total intensity for density calculation
        self.i_out_x_list = self.x[x_out_lower_index:x_out_lower_index+10]
        self.i_out_i_list = self.i[x_out_lower_index:x_out_lower_index+10]
        self.i_out = np.mean(self.i_out_i_list)

    def residuals_gauss(self,p,x,x_data):
        """Returns residuals for Gaussian fit of the intensity peak.
        Possible values for fit parameters are constrained to avoid
        overestimation of peak intensity.
        Args:
            p (list): fit parameters, [a, sigma, mu, c]
            x (list): position values
            x_data (list): intensity values
        Returns:
            residuals (list): residuals for fit
             -or-
            fail_array (list): in place of residuals if the fit fails
        """

        a, sigma, mu, c = p #unpacks p (for readability)

        i_peak_guess = gauss_func(p, mu)

        fail_array = np.ones(len(x)) * 99999.

        if all([sigma >= 0.1,
               abs(i_peak_guess - self.i[self.max_idx]) < 0.5 * self.i[self.max_idx]]):

            residuals = gauss_func(p,x) - x_data
            return residuals

        else:
            return fail_array

    def get_fwhm(self):
        """Calculates the full-width at half maximum (FWHM) of the linescan peak"""

        #determines half-max
        hm = (self.i_in + self.i_peak) / 2.
        # print hm

        # finds points closest to hm to the left of the peak
        search = self.i[:self.max_idx]
        self.left_index = (np.abs(search - hm)).argmin()
        if hm > self.i[self.left_index]:
            self.left_index_left = deepcopy(self.left_index)
            self.left_index_right = self.left_index_left + 1
        else:
            self.left_index_right = deepcopy(self.left_index)
            self.left_index_left = self.left_index_right - 1

        #gets interpolated intensity (linear interpolation between 2 surrounding points
        m_left = (self.i[self.left_index_right] - self.i[self.left_index_left]) /  (self.x[self.left_index_right] - self.x[self.left_index_left])
        b_left = self.i[self.left_index_right] - m_left * self.x[self.left_index_right]
        x_fwhm_left = (hm - b_left) / m_left
        self.fwhm_left = [x_fwhm_left,hm]

        #finds point closest to hm to the right of the peak
        search = self.i[self.max_idx:]
        self.right_index = (np.abs(search - hm)).argmin() + self.max_idx
        if hm < self.i[self.right_index]:
            self.right_index_left = deepcopy(self.right_index)
            self.right_index_right = self.right_index_left + 1
        else:
            self.right_index_right = deepcopy(self.right_index)
            self.right_index_left = self.right_index_right - 1

        #gets interpolated intensity (linear interpolation between 2 surrounding points
        m_right = (self.i[self.right_index_right] - self.i[self.right_index_left]) / (self.x[self.right_index_right] - self.x[self.right_index_left])
        b_right = self.i[self.right_index_right] - m_right * self.x[self.right_index_right]
        x_fwhm_right = (hm - b_right) / m_right
        self.fwhm_right = [x_fwhm_right,hm]

        self.fwhm = x_fwhm_right - x_fwhm_left

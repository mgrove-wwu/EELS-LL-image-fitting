#!/usr/bin/env python
# coding: utf-8

# # Preamble

# In[ ]:


#Loading Hyperspy Utilities and additional libraries
#from PySide import QtGui
#from PySide.QtGui import QMainWindow, QPushButton, QApplication

get_ipython().run_line_magic('matplotlib', 'qt5')
import numpy as np
import numpy.polynomial.polynomial as poly

from decimal import Decimal

import hyperspy
import hyperspy.api as hs

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg

from tqdm import tqdm, trange, tnrange, tqdm_notebook

import cv2
import sys
import os

from getch import getch, pause

from sympy import Symbol

from joblib import Parallel, delayed

import multiprocessing

from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfile, askdirectory

import mendeleev as mend
import sympy

import scipy
from scipy import ndimage as ndi
from scipy.signal import argrelextrema, correlate2d
from scipy.optimize import least_squares, curve_fit
import scipy.integrate as integrate

from mayavi import mlab

hs.preferences.gui(toolkit='ipywidgets')

#rc-Parameter definition
from matplotlib import rc

mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['figure.figsize'] = 10, 5
mpl.rcParams['figure.dpi'] = 200

plt.style.use('seaborn-whitegrid')
nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 12pt font in plots, to match 12pt font in document
        "axes.labelsize": 10,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
}
mpl.rcParams.update(nice_fonts)


# # main module / python class [EELS_image_fit] 
# 
# ### also includes sub routines that can be used but might lead to unexpected crashes

# In[2]:


# Class shall be used to map shear-transformation-zones for an image to visualize 
# shear-bands by considering energy-shifts of the plasmon-peaks aligned with the ZLP
class EELS_image_fit(object): 

    # Initialize the class for STZ-mapping 
    def __init__(self,
                 filename        = '', 
                 file            = None,
                 is_stack        = False, 
                 is_lazy         = False, 
                 binning         = True, 
                 includes_fits   = False,
                 deconv          = False,
                 RL_deconv_sigma = 0,
                 mkey            = None
                ):
        """
        EELS_image_fit: 
                    With careful analysis of an EELS-image the change of the plasmon energy due
                    to the change in topology can be investigated with additional thickness analysis,
                    Different outputs are possible to analyze the parameter shifts of the plasmon 
                    signature in the specimen.
                    
                    The minimum size of the spectrum image/list supported by this class is: 
                    (1-dim: 10 | 2-dim: 10*10=100) 
        ----------------------------------------------------------------------------------------------
        Initialization parameters:
                    filename: STRING      - specify filename and directory to load the EELS-image generated 
                                            (e.g. in GATAN)
                    file: Hyperspy-Signal - only use this when not using filename. Previously initia-
                                            lized Hyperspy Signals object. Read more on the Signals 
                                            objects http://hyperspy.org/hyperspy-doc/
                    is_stack: BOOLEAN     - loading multiple files by a wildcard "*"
                    
                    is_lazy: BOOLEAN      - to analyze big data (e.g. .dm4 - files) can be a problematic
                                            thing to due high memory requirements, therefore instead of
                                            using a numpy nd.array one can fall back to dask.dask_array
                                            resulting in an object which consists of multiple numpy arrays.
                                            This makes it possible to only work on parts of the datafile
                                            and therefore requiring only a small part of the memory require-
                                            ments.
                    binning: BOOLEAN      - operate on binned datasets. Due to caution set to FALSE as 
                                            default! Preferably, the datasets should only be binned using
                                            hyperspy to verify correct results.
                    RL_deconv_sigma: 0    - if this is set to a value other than 0, Richardson-Lucy deconv-
                                            deconvolution is used to estimate the unblurred spectrum 
                                            filtering out noise due to electron-photon-electron conversion,
                                            dark current image and the CCD readout process 
                                            [for more information see -> doi: 10.1016/S0304-3991(03)00103-7]
        """
        self.Filename          = ''
        self.File              = None
        self.is_lazy           = False
        
        self.File_deconv       = None
        self.attr_deconv       = deconv
        
        self.haadf             = None
        
        self.Fit_Model         = None
        
        self.all_models        = { '0' : 'VolumePlasmonDrude_leastsq_ls',
                                   '1' : 'VolumePlasmonDrude_mpfit_ls',
                                   '2' : 'VolumePlasmonDrude_NelderMead_ml',
                                   '3' : 'Lorentzian_leastsq_ls',
                                   '4' : 'Lorentzian_mpfit_ls',
                                   '5' : 'Lorentzian_NelderMead_ml',
                                   '6' : 'Gaussian_leastsq_ls',
                                   '7' : 'Gaussian_mpfit_ls',
                                   '8' : 'Gaussian_NelderMead_ml',
                                   '9' : 'Voigt_leastsq_ls',
                                   '10' : 'Voigt_mpfit_ls',
                                   '11' : 'Voigt_NelderMead_ml'
                                 }
        self.models_dict       = { }
        
        self.Chisq             = None
        self.red_Chisq         = None
        
        self.rchisq_mean       = None
        self.rchisq_std        = None
        
        self.eels_axis         = None
        self.function_set      = None
        self.model_name        = None
        self.optimizer         = None   
        self.method            = None
        
        self.ZLP_Emax          = None
        self.ZLP_FWHM          = None
        self.ZLP_Int           = None
        
        self.FPP_Emax          = None
        self.FPP_FWHM          = None
        self.FPP_Int           = None
        self.SPP_Emax          = None
        self.SPP_FWHM          = None
        self.SPP_Int           = None
        
        self.Ep_q0             = None
        
        self.param_dict        = {}
        
        self.linescans         = {}
        self.linescan_plots    = {}
        self.x0                = None
        self.x0_err            = None

        self.Elements          = None
        self.Concentrations    = None
        self.thickness_map     = None
        
        self.dir_list          = ['Plasmon characteristics', 
                                  'Thickness Evaluation'
                                 ]
        
        self.roi               = None
        self.roi_Ep            = None
        self.line              = None
        
        self.time              = 30
        
        self.elastic_threshold = None # arithmetic mean of the elastic threshold over the navigation space
        self.elastic_intensity = None
        
        self.load_data(filename, 
                       file, 
                       is_stack, 
                       is_lazy, 
                       binning,
                       RL_deconv_sigma,
                       mkey
                      )
        
    
    def check_underscores_in_title(self, 
                                   title,
                                   sep='_'
                                  ):
        """
        Titles cant have underscores outside of math environment for latex.
        This will check and if needed return the corrected title
        """
        title_res=''
        
        for string in title.split(sep=sep):
            title_res += string + ' '
        
        return title_res
        

    
    def print_file_information(self):
        """
        Printing standard metadata of file for beam and detector information as well as dataset information
        and the currently stored models of the file
        """
        if (self.attr_deconv == False):
            print(self.File.metadata)
            print(self.File.axes_manager)
            print(self.File.models)
        
        else:
            print(self.File_deconv.metadata)
            print(self.File_deconv.axes_manager)
            print(self.File_deconv.models)
            
        
    def yes_or_no(self, question):
        """
        Function for yes-no user input. Returns BOOLEAN value (y = True | n = False)
        """
        reply = str(input(question+' (y/n): ')).lower().strip()
        
        if (reply == ''):
            return self.yes_or_no("Pleas Enter")
        
        elif (reply[0] == 'y'):
            return True
        
        elif (reply[0] == 'n'):
            return False
        
        else:
            return self.yes_or_no("Please Enter")
        
    
    def detector_parameter_gui(self):
        """
        Comments missing
        """
        self.File.set_microscope_parameters()
    
    
    def model_gui(self, model):
        """
        Function to call the model attribute gui and await parameter adjustments until user is finished
        """
        model.gui()
        while True:
            finished = self.yes_or_no('When finished type (y) :')
            if (finished == True):
                break
        
    
    def split(self, txt, seps):
        """
        Splitting a string by specified seperators seps taking a tuple as input.
        """
        default_sep = seps[0]

        # we skip seps[0] because that's the default seperator
        for sep in seps[1:]:
            txt = txt.replace(sep, default_sep)
        return [i.strip() for i in txt.split(default_sep)]
    
    
    def richard_lucy_deconv(self, sigma):
        """
        Deconvolution using the Richardson-Lucy method using a gaussian 
        point spread function. Sigma is the standard deviation depending
        on the noise contributions. 
        If we assume that the effect of defocus for the different energy losses 
        is insignificant, then the PSF does not vary with the position 
        in the EELS detector. The electron–photon–electron conversion, 
        the dark current image and the CCD readout process may introduce 
        poissonian noise.
        The readout noise is Gaussian distributed.
        As the Poisson noise for larger sample sizes converges to a gaussian
        out of simplicity we assume an overall Gaussian approach for 
        deconvolution.
        For more information see:
            Improving energy resolution of EELS spectra: an alternative to the 
            monochromator solution
            
            A.Gloter,A.Douiri,M.Tencé,C.Colliex
        """
        size = self.File.axes_manager.signal_axes[0].size
        
        psf  = hs.signals.Signal1D(scipy.signal.gaussian(size, sigma, False))
        self.File.richardson_lucy_deconvolution(psf = psf)

        
    # Routine to load the data
    def load_data(self, 
                  filename, 
                  file, 
                  is_stack, 
                  is_lazy,
                  binning,
                  RL_deconv_sigma,
                  mkey
                 ):
        """
        load_data: This routine specifies the alignment of the space dimensions and
                   the zero-loss alignment as well as poissonian noise estimation and
                   optional model loading. Also it will consider a deconvolution by
                   the fourier log method (fully automated - still needs testing).
        ----------------------------------------------------------------------------------------------
        Initialization parameters:
                    filename: STRING - specify filename and directory to load the EELS-image generated 
                                       (e.g. in GATAN)
                    file: Hyperspy-Signal - only use this when not using filename. Previously initia-
                                            lized Hyperspy Signals object. Read more on the Signals 
                                            objects http://hyperspy.org/hyperspy-doc/
                    is_stack: BOOLEAN - loading multiple files by a wildcard "*"
                    
                    is_lazy: BOOLEAN - to analyze big data (e.g. .dm4 - files) can be a problematic
                                       thing to due high memory requirements, therefore instead of
                                       using a numpy nd.array one can fall back to dask.dask_array
                                       resulting in an object which consists of multiple numpy arrays.
                                       This makes it possible to only work on parts of the datafile
                                       and therefore requiring only a small part of the memory require-
                                       ments.
                                       
                    binning: BOOLEAN - should always be set to TRUE for EELS analysis
                                       as recommended by hyperspy's documentation.
                                       
                    deconv: BOOLEAN - specifies usage of the fourier log deconvolution
                                      to estimate the single scattered spectrum, isolating
                                      the plasmon peak without additional convolution.
                                      This can lead to better results, depending on the 
                                      quality of the SI and robustness of the fourier log
                                      method (worse results also possible).
        """
        self.is_lazy      = is_lazy
            
        if (filename    != ''):
            self.File     = hs.load(filename, stack = is_stack, lazy = is_lazy)

        elif (file        != None):
            if (is_lazy   == True):
                self.File = file.as_lazy()
            else:
                self.File = file
        
        else:
            Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
            filename      = askopenfilename()
            self.File     = hs.load(filename, stack = is_stack, lazy = is_lazy)

        self.Filename          = self.File.metadata.General.title
        
        axes                   = self.File.axes_manager.as_dictionary()
        for i in range(len(axes)):
            axis = axes['axis-' + str(i)]
            if (axis['units'] == 'eV'):
                self.eels_axis = self.File.axes_manager[i]

        including_fits = self.yes_or_no('Does the file include stored models from previous fitting?'
                                       )
        if (including_fits   == True):
                
            if (self.attr_deconv == True):
                self.File_deconv = self.File
                
            self.load_model(mkey=mkey)
        
            
            if (is_lazy == True):
                if (self.attr_deconv == False):
                    self.File.compute()
                    
                else:
                    self.File_deconv.compute()
                    
            self.elastic_threshold = np.nanmean(
                self.File.estimate_elastic_scattering_threshold().data
            )
            self.elastic_intensity = np.nanmean(
                self.File.estimate_elastic_scattering_intensity(self.elastic_threshold).data
            )
            self.File.metadata.Signal.binned = binning
            
            print('Aligning datastructure successful. Estimate poissonian noise...')
            self.File.estimate_poissonian_noise_variance()
        
        elif (including_fits == False):
            channels_before                     = self.File.axes_manager.signal_size
            self.File.data[self.File.data <= 0] = 1
            
            print('Setting up proper navigation space...')
            self.align_dataset()
            
            if (RL_deconv_sigma != 0):
                sigma = RL_deconv_sigma
                print('Using Richardson-Lucy-deconvolution to estimate ' +
                      'the unblurred spectrum using a standard deviation of ' +
                      str(sigma)
                     )
                self.richard_lucy_deconv(sigma)
            
            print('Aligning Zero-Loss Peak...')
            if (is_lazy == True):
                self.File.compute()
            
            self.File.align_zero_loss_peak(subpixel = True, signal_range=[-5.,5.])
        
            self.elastic_threshold = np.nanmean(
                self.File.estimate_elastic_scattering_threshold().data
            )
            self.elastic_intensity = np.nanmean(
                self.File.estimate_elastic_scattering_intensity(self.elastic_threshold).data
            )
            self.File.metadata.Signal.binned = binning
        
            print('Aligning datastructure successful. Estimate poissonian noise...')
            self.File.estimate_poissonian_noise_variance()
            
            if (self.attr_deconv == True):
                print('Calculating deconvoluted spectrum...')
                self.zlp_deconvolution()
            
            channels_after   = self.File.axes_manager.signal_size
            cropped_channels = channels_before - channels_after
            print('Loading process completed. Energy channels cropped by ZLP-Alignment: ' + 
                  str(cropped_channels) + '[' + str(cropped_channels/channels_before*100) + '%]' +
                  '\n' + 'Channels before: ' + str(channels_before) + '\n' +
                  'Channels after: ' + str(channels_after)
                 )
        
        if (is_lazy == True):
            self.File = self.File.as_lazy(copy_variance=True)
                
            if (self.attr_deconv == True):
                self.File_deconv = self.File_deconv.as_lazy(copy_variance=True)
            
        darkfield = self.yes_or_no('Do you want to load the corresponding darkfield image?\n' +
                                   '(Warning: The dimensions should match exactly for correct functionality.)'
                                  )
        if (darkfield == True):
            self.load_dfimage()
        
        
        print('Please check microscope and detector parameters,\n' + 
              'as the metadata from gatan might be incorrectly loaded\n' +
              'Hyperspy.'
             )
        
        if (self.attr_deconv == False):
            title      = self.File.metadata.General.title
            title_corr = self.check_underscores_in_title(title)

            self.File.metadata.General.title = title_corr
        
        else:
            title      = self.File_deconv.metadata.General.title
            title_corr = self.check_underscores_in_title(title)

            self.File_deconv.metadata.General.title = title_corr
        
        self.detector_parameter_gui()
        
    
    def load_dfimage(self, rotate=False):
        """
        Loading a corresponding dark field image into class variable for
        further processing (always using tk - file dialog)
        """
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        filename_haadf = askopenfilename()
        print('Loading ' + filename_haadf)
        self.haadf = hs.load(filename_haadf)
        if rotate == True:
            self.haadf = hs.signals.Signal2D(np.transpose(self.haadf.data))
    
    
    def align_dataset(self):
        """
        align_dataset: By dimension the dataset will be devided into different objects as follows:
                           EELS: 1D array specifying Counts/Intensity along the energy loss axis
                                 given by offset and gain as defined in the metadata
                           EELS-stack: 2D array specifying Counts/Intensity as above (see Object EELS)
                                       and 1 navigation axis
                           EELS-image: 3D array specifying Counts/Intensity as above (see Object EELS)
                                       and 2 navigation axis - therefore the name... :)
        ----------------------------------------------------------------------------------------------
        Initialization parameters:
                    None, all parameters should have been included in the metadata by GATAN or by
                    defining the meta information in hyperspy yourself.
        """
        self.File = self.File.as_signal1D(spectral_axis=self.eels_axis)
        
        self.File.set_signal_type("EELS")
        self.File.axes_manager.signal_axes[0].name = 'Energy loss'
        
        for axis in self.File.axes_manager.navigation_axes:
            axis.offset = 0
            axis.scale  = abs(axis.scale)

    
    def zlp_deconvolution(self, lazy):
        """
        Uses Fourier-Log method for deconvolution and elastic scattering threshold to estimate ZLP.
        """
        print(self.elastic_threshold)
        zlp              = self.File.isig[:self.elastic_threshold]
        self.File_deconv = self.File.fourier_log_deconvolution(zlp)
        
    
    def init_func(self):
        """
        Initialize distribution type for ZLP, FPP, SPP. Three types are currently supported:
        func: {'VolumePlasmonDrude', 'Lorentzian', 'Gaussian', 'Voigt'}
        """
        param_init = self.init_model_params()
        
        zlp_pos    = param_init[0]
        zlp_fwhm   = param_init[1]
        zlp_int    = param_init[2]
        fpp_pos    = param_init[3]
        fpp_fwhm   = param_init[4]
        fpp_int    = param_init[5]
        spp_pos    = param_init[6]
        spp_fwhm   = param_init[7]
        spp_int    = param_init[8]
        
        try:
            if (self.function_set == 'VolumePlasmonDrude'):

                func_1 = hs.model.components1D.Voigt()
                func_1.area.value                = zlp_int
                func_1.centre.value              = zlp_pos
                func_1.FWHM.value                = zlp_fwhm / 2
                func_1.gamma.value               = zlp_fwhm / 2
                func_1.name = 'Zero_Loss_Peak'
                
                func_2 = hs.model.components1D.VolumePlasmonDrude(intensity      = fpp_int,
                                                                  plasmon_energy = fpp_pos,
                                                                  fwhm           = fpp_fwhm
                                                                 )
                func_2.name = 'First_Plasmon_Peak'

                func_3 = hs.model.components1D.VolumePlasmonDrude(intensity      = spp_int,
                                                                  plasmon_energy = spp_pos,
                                                                  fwhm           = spp_fwhm
                                                                 )
                func_3.name = 'Second_Plasmon_Peak'
                
            elif (self.function_set == 'Lorentzian'):

                func_1 = hs.model.components1D.Voigt()
                func_1.area.value                = zlp_int
                func_1.centre.value              = zlp_pos
                func_1.FWHM.value                = zlp_fwhm / 2
                func_1.gamma.value               = zlp_fwhm / 2
                func_1.name = 'Zero_Loss_Peak'
                
                func_2 = hs.model.components1D.Lorentzian(A      = fpp_int, 
                                                          centre = fpp_pos,
                                                          gamma  = fpp_fwhm / 2
                                                         )
                func_2.name = 'First_Plasmon_Peak'

                func_3 = hs.model.components1D.Lorentzian(A      = spp_int, 
                                                          centre = spp_pos,
                                                          gamma  = spp_fwhm / 2
                                                         )
                func_3.name = 'Second_Plasmon_Peak'
                
            elif (self.function_set == 'Gaussian'):

                func_1 = hs.model.components1D.Voigt()
                func_1.area.value                = zlp_int
                func_1.centre.value              = zlp_pos
                func_1.FWHM.value                = zlp_fwhm / 2
                func_1.gamma.value               = zlp_fwhm / 2
                func_1.name = 'Zero_Loss_Peak'

                func_2 = hs.model.components1D.Gaussian(A      = fpp_int,
                                                        centre = fpp_pos,
                                                        sigma  = fpp_fwhm / (2 * np.sqrt( np.log(2) ))
                                                       )
                func_2.name = 'First_Plasmon_Peak'

                func_3 = hs.model.components1D.Gaussian(A      = spp_int,
                                                        centre = spp_pos,
                                                        sigma  = spp_fwhm / (2 * np.sqrt( np.log(2) ))
                                                       )
                func_3.name = 'Second_Plasmon_Peak'

            elif (self.function_set == 'Voigt'):

                func_1 = hs.model.components1D.Voigt()
                func_1.area.value                = zlp_int
                func_1.centre.value              = zlp_pos
                func_1.FWHM.value                = zlp_fwhm / 2
                func_1.gamma.value               = zlp_fwhm / 2
                func_1.name = 'Zero_Loss_Peak'
                
                func_2 = hs.model.components1D.Voigt()
                func_2.area.value   = fpp_int
                func_2.centre.value = fpp_pos
                func_2.FWHM.value   = fpp_fwhm / 2
                func_2.gamma.value  = fpp_fwhm / 2
                func_2.name = 'First_Plasmon_Peak'
                
                func_3 = hs.model.components1D.Voigt()
                func_2.area.value   = spp_int
                func_2.centre.value = spp_pos
                func_2.FWHM.value   = spp_fwhm / 2
                func_2.gamma.value  = spp_fwhm / 2
                func_3.name = 'Second_Plasmon_Peak'
            
            return func_1, func_2, func_3
            
        except:
            print('No distributions initialized. Please try again. For more information, see docstring.')
        

    
    # Function to fit EELS-Spectra with three gaussian
    # Another option is to use code in 2nd cell to use extra class
    def eels_fit_routine(self, 
                         function_set='VolumePlasmonDrude', 
                         fitter='leastsq', 
                         method='ls', 
                         samfire=False, 
                         multithreading=False, 
                         workers=4,
                         auto=True
                        ):         
        """
        Routine to initialize fit routine. Storing the calculated model in the loaded file.
        
        Initialize distribution model for ZLP, FPP, SPP. Three types are currently supported:
        function_Set: {'VolumePlasmonDrude', 'Lorentzian', 'Gaussian', 'Voigt'}
        """
        # Initialize serial fitting routine
        self.function_set = function_set
        self.model_name = function_set + str('_') + fitter + str('_') + method
        
        if (samfire == False):

            self.Fit_Model = self.fit_eels(fitter, 
                                           method,
                                           auto,
                                           self.is_lazy
                                          )

        # Initialize SAMFire (parallel) fitting routine
        else:

            self.Fit_Model = self.fit_eels_SAMF(fitter, 
                                                method, 
                                                multithreading, 
                                                workers,
                                                auto,
                                                self.is_lazy
                                               )
        
        print('Model will be stored in file...')
        self.Fit_Model.store(name = self.model_name)
        self.update_models_dict()
        
        print('Stored models in file:')
        if (self.attr_deconv == False):
            print(self.File.models)

        elif (self.attr_deconv == True):
            print(self.File_deconv.models)
            
        self.generate_param_maps(method)
    
    
    def init_model_params(self):
        """
        Comments missing
        """
        if (self.attr_deconv == False):
            print('Estimate function parameters...')

            mean            = self.File.mean()
            
            if (self.is_lazy == True):
                mean.compute()

            # estimate peak parameters for the zero loss peak
            elastic         = mean.isig[:self.elastic_threshold]

            # as ohaver implementation only looks for peaks in the positive energy 
            # range, the axis will be shifted forward for this calculation.
            # the shift will be considered for the peak position afterwards,
            # as it behaves linear.
            offset          = elastic.axes_manager.signal_axes[0].offset
            elastic.axes_manager.signal_axes[0].offset -= offset

            axis_elastic    = elastic.axes_manager.signal_axes[0]
            amp_zlp         = np.max(elastic, axis=axis_elastic).data[0]
            param_elastic   = elastic.find_peaks1D_ohaver(amp_thresh = 0.1*amp_zlp,
                                                          maxpeakn   = 1
                                                         )

            if (len(param_elastic) == 1):
                zlp_pos     = param_elastic[0][0][0] + offset
                zlp_fwhm    = param_elastic[0][0][2]

                zlp_int     = self.elastic_intensity

            else:
                zlp_pos     = 0. # in eV
                zlp_fwhm    = 1. # in eV
                zlp_int     = self.elastic_intensity

            inelastic       = mean.isig[self.elastic_threshold:]

            axis_inelastic  = inelastic.axes_manager.signal_axes[0]
            amp_pp          = np.max(inelastic, axis=axis_inelastic).data[0]

            param_inelastic = inelastic.find_peaks1D_ohaver(amp_thresh = 0.1*amp_pp,
                                                            maxpeakn   = 2
                                                           )

            if (len(param_inelastic) >= 1):
                fpp_pos     = param_inelastic[0][0][0]
                fpp_fwhm    = param_inelastic[0][0][2]

                if (self.function_set == 'VolumePlasmonDrude'):
                    fpp_int = amp_pp

                else:
                    FPP_signal  = inelastic.isig[fpp_pos - fpp_fwhm : fpp_pos + fpp_fwhm] 
                    fpp_axis    = FPP_signal.axes_manager.signal_axes[0]
                    fpp_int     = (1 / scipy.special.erf( np.sqrt( np.log(4) ) ) * 
                                   FPP_signal.integrate1D(fpp_axis).data[0]
                                  )

            else:
                print('Could not estimate initial parameters. Results could diverge!' +
                      'It is highly recommended to adjust the parameters manually.'
                     )
                fpp_pos     = 15. # in eV
                fpp_fwhm    = 2.  # in eV

                if (self.function_set == 'VolumePlasmonDrude'):
                    fpp_int = amp_pp

                else:
                    FPP_signal  = inelastic.isig[fpp_pos - fpp_fwhm : fpp_pos + fpp_fwhm] 
                    fpp_axis    = FPP_signal.axes_manager.signal_axes[0]
                    fpp_int     = (1 / scipy.special.erf( np.sqrt( np.log(4) ) ) * 
                                   FPP_signal.integrate1D(fpp_axis).data[0]
                              )

            if (len(param_inelastic) == 2):
                spp_pos     = param_inelastic[0][1][0]
                spp_fwhm    = param_inelastic[0][1][2]

                if (self.function_set == 'VolumePlasmonDrude'):
                    spp_int = amp_pp * 0.3

                else:
                    SPP_signal  = inelastic.isig[spp_pos - spp_fwhm : spp_pos + spp_fwhm] 
                    spp_axis    = SPP_signal.axes_manager.signal_axes[0]
                    spp_int     = (1 / scipy.special.erf( np.sqrt( np.log(4) ) ) * 
                                   SPP_signal.integrate1D(spp_axis).data[0]
                                  )

            else:
                spp_pos     = 2 * fpp_pos
                spp_fwhm    = 2 * fpp_fwhm

                if (self.function_set == 'VolumePlasmonDrude'):
                    spp_int = amp_pp * 0.3

                else:
                    SPP_signal  = inelastic.isig[spp_pos - spp_fwhm : spp_pos + spp_fwhm] 
                    spp_axis    = SPP_signal.axes_manager.signal_axes[0]
                    spp_int     = (1 / scipy.special.erf( np.sqrt( np.log(4) ) ) * 
                                   SPP_signal.integrate1D(spp_axis).data[0]
                                  )

            if (fpp_int > zlp_int or spp_int > zlp_int):
                print('Attention: The intensity of one of the Plasmon peaks is\n' + 
                      '           higher than the Zero-loss intensity!\n' + 
                      '           The model should be monitored carefully.\n' +
                      '           It is recommended to gather a new spectrum of\n' +
                      '           a sample with smaller thickness to minimize\n' +
                      '           the inelastically scattered part of the spectrum!'
                     )

            return zlp_pos, zlp_fwhm, zlp_int, fpp_pos, fpp_fwhm, fpp_int, spp_pos, spp_fwhm, spp_int
        
        else:
            print('Estimate function parameters...')

            mean            = self.File.mean()
            if (self.is_lazy == True):
                mean.compute()

            # estimate peak parameters for the zero loss peak
            elastic         = mean.isig[:a.elastic_threshold]

            # as ohaver implementation only looks for peaks in the positive energy 
            # range, the axis will be shifted forward for this calculation.
            # the shift will be considered for the peak position afterwards,
            # as it behaves linear.
            offset          = elastic.axes_manager.signal_axes[0].offset
            elastic.axes_manager.signal_axes[0].offset -= offset

            axis_elastic    = elastic.axes_manager.signal_axes[0]
            amp_zlp         = 0
            zlp_pos     = 0 + offset
            zlp_fwhm    = 1

            zlp_int     = 0

            inelastic       = mean.isig[self.elastic_threshold:]

            axis_inelastic  = inelastic.axes_manager.signal_axes[0]
            amp_pp          = np.max(inelastic, axis=axis_inelastic).data[0]

            param_inelastic = inelastic.find_peaks1D_ohaver(amp_thresh = 0.1*amp_pp,
                                                            maxpeakn   = 2
                                                           )
            
            if (len(param_inelastic) >= 1):
                fpp_pos     = param_inelastic[0][0][0]
                fpp_fwhm    = param_inelastic[0][0][2]
                    
                if (fpp_pos < self.elastic_threshold or fpp_pos > 50.):    
                    fpp_pos     = 15.
                    fpp_fwhm    = 2.

                if (self.function_set == 'VolumePlasmonDrude'):
                    fpp_int     = amp_pp

                else:
                    FPP_signal  = inelastic.isig[fpp_pos - fpp_fwhm : fpp_pos + fpp_fwhm] 
                    fpp_axis    = FPP_signal.axes_manager.signal_axes[0]
                    fpp_int     = (1 / scipy.special.erf( np.sqrt( np.log(4) ) ) * 
                                   FPP_signal.integrate1D(fpp_axis).data[0]
                                  )

            else:
                print('Could not estimate initial parameters. Results could diverge!' +
                      'It is highly recommended to adjust the parameters manually.'
                     )
                fpp_pos     = 15. # in eV
                fpp_fwhm    = 2.  # in eV

                if (self.function_set == 'VolumePlasmonDrude'):
                    fpp_int     = amp_pp

                else:
                    FPP_signal  = inelastic.isig[fpp_pos - fpp_fwhm : fpp_pos + fpp_fwhm] 
                    fpp_axis    = FPP_signal.axes_manager.signal_axes[0]
                    fpp_int     = (1 / scipy.special.erf( np.sqrt( np.log(4) ) ) * 
                                   FPP_signal.integrate1D(fpp_axis).data[0]
                              )

            if (len(param_inelastic) == 2):
                spp_pos     = param_inelastic[0][1][0]
                spp_fwhm    = param_inelastic[0][1][2]
                    
                if (spp_pos <= fpp_pos or spp_pos > 75.):
                    spp_pos     = 2 * fpp_pos
                    spp_fwhm    = 2 * fpp_fwhm

                if (self.function_set == 'VolumePlasmonDrude'):
                    spp_int = amp_pp * 0.3

                else:
                    SPP_signal  = inelastic.isig[spp_pos - spp_fwhm : spp_pos + spp_fwhm] 
                    spp_axis    = SPP_signal.axes_manager.signal_axes[0]
                    spp_int     = (1 / scipy.special.erf( np.sqrt( np.log(4) ) ) * 
                                   SPP_signal.integrate1D(spp_axis).data[0]
                                  )

            else:
                spp_pos     = 2 * fpp_pos
                spp_fwhm    = 2 * fpp_fwhm

                if (self.function_set == 'VolumePlasmonDrude'):
                    spp_int = 0

                else:
                    spp_int     = 0

            return zlp_pos, zlp_fwhm, zlp_int, fpp_pos, fpp_fwhm, fpp_int, spp_pos, spp_fwhm, spp_int
        
        
    def init_model(self, 
                   mean = False
                  ):
        """
        Comments missing
        """
        if (self.attr_deconv == False):
                
            offset = self.File.axes_manager['Energy loss'].offset
            scale  = self.File.axes_manager['Energy loss'].scale
            size   = self.File.axes_manager['Energy loss'].size

            e_max  = float(int(scale * size + offset))
            
            if (mean == True):
                model               = self.File.mean().create_model(
                    auto_background = False
                )

            elif (mean == False):
                model               = self.File.create_model(
                    auto_background = False
                )
                
            func1, func2, func3 = self.init_func()
            params              = self.init_model_params()

            model.set_signal_range(-self.elastic_threshold, 
                                   e_max
                                  )
            model.extend([func1, func2, func3])

            self.set_model_params(model, 
                                  params = params, 
                                  func   = 'Zero_Loss_Peak'
                                 )
            self.set_model_params(model, 
                                  params = params, 
                                  func   = 'First_Plasmon_Peak'
                                 )
            self.set_model_params(model, 
                                  params = params, 
                                  func   = 'Second_Plasmon_Peak'
                                 )

            self.set_second_plasmonenergy(model)

            return model, params
                
        else:
                
            offset = self.File_deconv.axes_manager['Energy loss'].offset
            scale  = self.File_deconv.axes_manager['Energy loss'].scale
            size   = self.File_deconv.axes_manager['Energy loss'].size

            e_max  = float(int(scale * size + offset))
        
            if (mean == True):
                model               = self.File_deconv.mean().create_model(
                    auto_background = False
                )

            elif (mean == False):
                model               = self.File_deconv.create_model(
                    auto_background = False
                )

            func1, func2, func3 = self.init_func()
            params              = self.init_model_params()
            
            # disable n-th order scattering for n=2 (set SPP-parameters to 0), 
            # larger order scattering should not occur due to sample thickness
            # recommended for TEM analysis
            # as Tuples cannot be accessed by indexing, transformation for
            # Tuple -> List -> Tuple is needed
            params_SPP_contrib_0 = list(params)
            params_SPP_contrib_0[6]          = 0
            params_SPP_contrib_0[7]          = 0
            params_SPP_contrib_0[8]          = 0
            params                           = tuple(params_SPP_contrib_0)
            
            model.set_signal_range(self.elastic_threshold, 
                                   e_max
                                  )
            model.extend([func1, func2, func3])

            self.set_model_params(model, 
                                  params = params, 
                                  func   = 'Zero_Loss_Peak'
                                 )
            self.set_model_params(model, 
                                  params = params, 
                                  func   = 'First_Plasmon_Peak'
                                 )
            self.set_model_params(model, 
                                  params = params, 
                                  func   = 'Second_Plasmon_Peak'
                                 )

            self.set_second_plasmonenergy(model)

            return model, params
    
    
    def set_second_plasmonenergy(self, model):
        """
        Comments missing
        """
        if (self.function_set != 'VolumePlasmonDrude'):
            model.components.Second_Plasmon_Peak.centre.twin_function_expr         = '2*x'
            model.components.Second_Plasmon_Peak.centre.inverse_twin_function_expr = 'x/2'
            model.components.Second_Plasmon_Peak.centre.twin                       = model.components.First_Plasmon_Peak.centre

        else:
            model.components.Second_Plasmon_Peak.plasmon_energy.twin_function_expr          = '2*x'
            model.components.Second_Plasmon_Peak.plasmon_energy.inverse_twin_function_expr  = 'x/2'
            model.components.Second_Plasmon_Peak.plasmon_energy.twin                        = model.components.First_Plasmon_Peak.plasmon_energy
    
    
    def fit_zlp_only(self, 
                     model, 
                     fitter, 
                     method,
                     bounded
                    ):
        """
        Comments missing
        """
        mean, params = self.init_model(mean = True)
        
        self.set_model_params(model, 
                              params = params, 
                              func   = 'Zero_Loss_Peak'
                             )
        
        self.set_bounds(mean, 
                        bounded
                       )
        
        mean.components.Zero_Loss_Peak.active      = True
        mean.components.First_Plasmon_Peak.active  = False
        mean.components.Second_Plasmon_Peak.active = False

        zlp_pos             = params[0]
        zlp_fwhm            = params[1]
        
        mean.set_signal_range(zlp_pos - zlp_fwhm, 
                              zlp_pos + zlp_fwhm
                             )
        
        mean.fit(fitter  = fitter, 
                 method  = method,
                 bounded = bounded
                )
        
        self.set_model_params(model, 
                              mean = mean, 
                              func = 'Zero_Loss_Peak'
                             )
        
        
    def fit_pp_only(self,
                    upper_bound,
                    model, 
                    fitter, 
                    method,
                    bounded
                   ):
        """
        Comments missing
        """
        mean, params = self.init_model(mean = True)
        
        self.set_model_params(model, 
                              params = params, 
                              func   = 'First_Plasmon_Peak'
                             )
        
        self.set_model_params(model, 
                              params = params, 
                              func   = 'Second_Plasmon_Peak'
                             )
        
        self.set_bounds(mean, 
                        bounded
                       )
        
        mean.components.Zero_Loss_Peak.active      = False
        mean.components.First_Plasmon_Peak.active  = True
        mean.components.Second_Plasmon_Peak.active = True

        self.set_second_plasmonenergy(mean)
        
        # used for the signal range to exclude additional influence
        fpp_pos  = params[3]
        fpp_fwhm = params[4]
        spp_pos  = params[6]
        spp_fwhm = params[7]
        
        mean.set_signal_range(fpp_pos - fpp_fwhm, 
                              spp_pos + spp_fwhm
                             )

        mean.fit(fitter=fitter, 
                 method=method,
                 bounded=bounded
                )
        
        self.set_model_params(model, 
                              mean = mean, 
                              func = 'First_Plasmon_Peak'
                             )
        
        self.set_model_params(model, 
                              mean = mean, 
                              func = 'Second_Plasmon_Peak'
                             )
        
        return self.get_plasmonenergy(mean)
        
    
    def fit_fpp_only(self, 
                     upper_bound,
                     model, 
                     fitter, 
                     method,
                     bounded
                    ):
        """
        Comments missing
        """
        mean, params = self.init_model(mean = True)
        
        self.set_bounds(mean, 
                        bounded
                       )
        
        mean.components.Zero_Loss_Peak.active      = False
        mean.components.First_Plasmon_Peak.active  = True
        mean.components.Second_Plasmon_Peak.active = False

        mean.set_signal_range(self.elastic_threshold, #see above
                              upper_bound #eV
                             )
        
        mean.fit(fitter=fitter, 
                 method=method,
                 bounded=bounded
                )
        
        plasmonenergy = self.get_plasmonenergy(mean)
        
        mean.set_signal_range(self.elastic_threshold, #see above
                              2.5 * plasmonenergy 
                              #using better estimate of plasmon energy range to increase accuracy 
                             )

        mean.fit(fitter  = fitter, 
                 method  = method,
                 bounded = bounded
                )
        
        self.set_model_params(model, 
                              mean = mean, 
                              func = 'First_Plasmon_Peak'
                             )
        
        return self.get_plasmonenergy(mean)
        
    
    def get_plasmonenergy(self,
                          mean
                         ):
        """
        Comments missing
        """
        if (self.function_set == 'VolumePlasmonDrude'):
            plasmonenergy = mean.components.First_Plasmon_Peak.plasmon_energy.value
        
        else:
            plasmonenergy = mean.components.First_Plasmon_Peak.centre.value
            
        return plasmonenergy
    
    
    def set_bounds(self, 
                   model, 
                   bounded
                  ):
        """
        Comments missing
        """
        if (self.function_set == 'Voigt'):
            
            if (bounded):
                model.components.Zero_Loss_Peak.area.bmin   = (self.elastic_intensity 
                                                               - self.elastic_intensity / 2
                                                              )
                model.components.Zero_Loss_Peak.area.bmax   = (self.elastic_intensity 
                                                               + self.elastic_intensity / 2
                                                              )
                model.components.Zero_Loss_Peak.centre.bmin = -self.elastic_threshold
                model.components.Zero_Loss_Peak.centre.bmax = self.elastic_threshold
                model.components.Zero_Loss_Peak.FWHM.bmin   = 0.
                model.components.Zero_Loss_Peak.FWHM.bmax   = 2 * self.elastic_threshold
                model.components.Zero_Loss_Peak.gamma.bmin  = 0.
                model.components.Zero_Loss_Peak.gamma.bmax  = 2 * self.elastic_threshold


                model.components.First_Plasmon_Peak.area.bmin   = 0.
                model.components.First_Plasmon_Peak.area.bmax   = (self.elastic_intensity 
                                                                   + self.elastic_intensity / 2
                                                                  )
                model.components.First_Plasmon_Peak.centre.bmin = self.elastic_threshold
                model.components.First_Plasmon_Peak.centre.bmax = 50.
                model.components.First_Plasmon_Peak.FWHM.bmin   = 0.
                model.components.First_Plasmon_Peak.FWHM.bmax   = 50.
                model.components.First_Plasmon_Peak.gamma.bmin  = 0.
                model.components.First_Plasmon_Peak.gamma.bmax  = 50.


                model.components.Second_Plasmon_Peak.area.bmin   = 0.
                model.components.Second_Plasmon_Peak.area.bmax   = (self.elastic_intensity 
                                                                    + self.elastic_intensity / 2
                                                                   )
                model.components.Second_Plasmon_Peak.centre.bmin = self.elastic_threshold
                model.components.Second_Plasmon_Peak.centre.bmax = 100.
                model.components.Second_Plasmon_Peak.FWHM.bmin   = 0.
                model.components.Second_Plasmon_Peak.FWHM.bmax   = 100.
                model.components.Second_Plasmon_Peak.gamma.bmin  = 0.
                model.components.Second_Plasmon_Peak.gamma.bmax  = 100.
                
            
            else:
                model.components.Zero_Loss_Peak.area.ext_force_positive        = True
                model.components.Zero_Loss_Peak.centre.ext_force_positive      = False
                model.components.Zero_Loss_Peak.FWHM.ext_force_positive        = True
                model.components.Zero_Loss_Peak.gamma.ext_force_positive       = True
                model.components.First_Plasmon_Peak.area.ext_force_positive    = True
                model.components.First_Plasmon_Peak.centre.ext_force_positive  = True
                model.components.First_Plasmon_Peak.FWHM.ext_force_positive    = True
                model.components.First_Plasmon_Peak.gamma.ext_force_positive   = True
                model.components.Second_Plasmon_Peak.area.ext_force_positive   = True
                model.components.Second_Plasmon_Peak.centre.ext_force_positive = True
                model.components.Second_Plasmon_Peak.FWHM.ext_force_positive   = True
                model.components.Second_Plasmon_Peak.gamma.ext_force_positive  = True
        
        
        elif (self.function_set == 'Lorentzian'):
            
            if (bounded):
                model.components.Zero_Loss_Peak.area.bmin   = (self.elastic_intensity 
                                                               - self.elastic_intensity / 2
                                                              )
                model.components.Zero_Loss_Peak.area.bmax   = (self.elastic_intensity 
                                                               + self.elastic_intensity / 2
                                                              )
                model.components.Zero_Loss_Peak.centre.bmin = -self.elastic_threshold
                model.components.Zero_Loss_Peak.centre.bmax = self.elastic_threshold
                model.components.Zero_Loss_Peak.FWHM.bmin   = 0.
                model.components.Zero_Loss_Peak.FWHM.bmax   = 2 * self.elastic_threshold
                model.components.Zero_Loss_Peak.gamma.bmin  = 0.
                model.components.Zero_Loss_Peak.gamma.bmax  = 2 * self.elastic_threshold


                model.components.First_Plasmon_Peak.A.bmin      = 0.
                model.components.First_Plasmon_Peak.A.bmax      = (self.elastic_intensity 
                                                                   + self.elastic_intensity / 2
                                                                  )
                model.components.First_Plasmon_Peak.centre.bmin = self.elastic_threshold
                model.components.First_Plasmon_Peak.centre.bmax = 50.
                model.components.First_Plasmon_Peak.gamma.bmin  = 0.
                model.components.First_Plasmon_Peak.gamma.bmax  = 50.


                model.components.Second_Plasmon_Peak.A.bmin      = 0.
                model.components.Second_Plasmon_Peak.A.bmax      = (self.elastic_intensity 
                                                                    + self.elastic_intensity / 2
                                                                   )
                model.components.Second_Plasmon_Peak.centre.bmin = self.elastic_threshold
                model.components.Second_Plasmon_Peak.centre.bmax = 100.
                model.components.Second_Plasmon_Peak.gamma.bmin  = 0.
                model.components.Second_Plasmon_Peak.gamma.bmax  = 100.

            
            else:
                model.components.Zero_Loss_Peak.area.ext_force_positive        = True
                model.components.Zero_Loss_Peak.centre.ext_force_positive      = False
                model.components.Zero_Loss_Peak.FWHM.ext_force_positive        = True
                model.components.Zero_Loss_Peak.gamma.ext_force_positive       = True
                model.components.First_Plasmon_Peak.A.ext_force_positive       = True
                model.components.First_Plasmon_Peak.centre.ext_force_positive  = True
                model.components.First_Plasmon_Peak.gamma.ext_force_positive   = True
                model.components.Second_Plasmon_Peak.A.ext_force_positive      = True
                model.components.Second_Plasmon_Peak.centre.ext_force_positive = True
                model.components.Second_Plasmon_Peak.gamma.ext_force_positive  = True
                
                
        elif (self.function_set == 'Gaussian'):
            
            if (bounded):
                model.components.Zero_Loss_Peak.area.bmin   = (self.elastic_intensity 
                                                               - self.elastic_intensity / 2
                                                              )
                model.components.Zero_Loss_Peak.area.bmax   = (self.elastic_intensity 
                                                               + self.elastic_intensity / 2
                                                              )
                model.components.Zero_Loss_Peak.centre.bmin = -self.elastic_threshold
                model.components.Zero_Loss_Peak.centre.bmax = self.elastic_threshold
                model.components.Zero_Loss_Peak.FWHM.bmin   = 0.
                model.components.Zero_Loss_Peak.FWHM.bmax   = 2 * self.elastic_threshold
                model.components.Zero_Loss_Peak.gamma.bmin  = 0.
                model.components.Zero_Loss_Peak.gamma.bmax  = 2 * self.elastic_threshold


                model.components.First_Plasmon_Peak.A.bmin      = 0.
                model.components.First_Plasmon_Peak.A.bmax      = (self.elastic_intensity 
                                                                   + self.elastic_intensity / 2
                                                                  )
                model.components.First_Plasmon_Peak.centre.bmin = self.elastic_threshold
                model.components.First_Plasmon_Peak.centre.bmax = 50.
                model.components.First_Plasmon_Peak.sigma.bmin  = 0.
                model.components.First_Plasmon_Peak.sigma.bmax  = 50.


                model.components.Second_Plasmon_Peak.A.bmin      = 0.
                model.components.Second_Plasmon_Peak.A.bmax      = (self.elastic_intensity 
                                                                    + self.elastic_intensity / 2
                                                                   )
                model.components.Second_Plasmon_Peak.centre.bmin = self.elastic_threshold
                model.components.Second_Plasmon_Peak.centre.bmax = 100.
                model.components.Second_Plasmon_Peak.sigma.bmin  = 0.
                model.components.Second_Plasmon_Peak.sigma.bmax  = 100.
            

            else:
                model.components.Zero_Loss_Peak.area.ext_force_positive        = True
                model.components.Zero_Loss_Peak.centre.ext_force_positive      = False
                model.components.Zero_Loss_Peak.FWHM.ext_force_positive        = True
                model.components.Zero_Loss_Peak.gamma.ext_force_positive       = True
                model.components.First_Plasmon_Peak.A.ext_force_positive       = True
                model.components.First_Plasmon_Peak.centre.ext_force_positive  = True
                model.components.First_Plasmon_Peak.sigma.ext_force_positive   = True
                model.components.Second_Plasmon_Peak.A.ext_force_positive      = True
                model.components.Second_Plasmon_Peak.centre.ext_force_positive = True
                model.components.Second_Plasmon_Peak.sigma.ext_force_positive  = True
                
            
        elif (self.function_set == 'VolumePlasmonDrude'):
            
            if (bounded):
                model.components.Zero_Loss_Peak.area.bmin   = (self.elastic_intensity 
                                                               - self.elastic_intensity / 2
                                                              )
                model.components.Zero_Loss_Peak.area.bmax   = (self.elastic_intensity 
                                                               + self.elastic_intensity / 2
                                                              )
                model.components.Zero_Loss_Peak.centre.bmin = -self.elastic_threshold
                model.components.Zero_Loss_Peak.centre.bmax = self.elastic_threshold
                model.components.Zero_Loss_Peak.FWHM.bmin   = 0.
                model.components.Zero_Loss_Peak.FWHM.bmax   = 2 * self.elastic_threshold
                model.components.Zero_Loss_Peak.gamma.bmin  = 0.
                model.components.Zero_Loss_Peak.gamma.bmax  = 2 * self.elastic_threshold


                model.components.First_Plasmon_Peak.intensity.bmin      = 0.
                model.components.First_Plasmon_Peak.intensity.bmax      = (self.elastic_intensity 
                                                                           + self.elastic_intensity / 2
                                                                          )
                model.components.First_Plasmon_Peak.plasmon_energy.bmin = self.elastic_threshold
                model.components.First_Plasmon_Peak.plasmon_energy.bmax = 50.
                model.components.First_Plasmon_Peak.fwhm.bmin  = 0.
                model.components.First_Plasmon_Peak.fwhm.bmax  = 50.


                model.components.Second_Plasmon_Peak.intensity.bmin      = 0.
                model.components.Second_Plasmon_Peak.intensity.bmax      = (self.elastic_intensity 
                                                                            + self.elastic_intensity / 2
                                                                           )
                model.components.Second_Plasmon_Peak.plasmon_energy.bmin = self.elastic_threshold
                model.components.Second_Plasmon_Peak.plasmon_energy.bmax = 100.
                model.components.Second_Plasmon_Peak.fwhm.bmin  = 0.
                model.components.Second_Plasmon_Peak.fwhm.bmax  = 100.


            else:
                model.components.Zero_Loss_Peak.area.ext_force_positive                = True
                model.components.Zero_Loss_Peak.centre.ext_force_positive              = False
                model.components.Zero_Loss_Peak.FWHM.ext_force_positive                = True
                model.components.Zero_Loss_Peak.gamma.ext_force_positive               = True
                model.components.First_Plasmon_Peak.intensity.ext_force_positive       = True
                model.components.First_Plasmon_Peak.plasmon_energy.ext_force_positive  = True
                model.components.First_Plasmon_Peak.fwhm.ext_force_positive            = True
                model.components.Second_Plasmon_Peak.intensity.ext_force_positive      = True
                model.components.Second_Plasmon_Peak.plasmon_energy.ext_force_positive = True
                model.components.Second_Plasmon_Peak.fwhm.ext_force_positive           = True
                
    
    def set_by_params(self,
                      model,
                      params,
                      func
                     ):
        """
        Comments missing
        """
        func_dict = {'Zero_Loss_Peak'      : params[0:3],
                     'First_Plasmon_Peak'  : params[3:6],
                     'Second_Plasmon_Peak' : params[6:9]
                    }

        if (self.function_set == 'Voigt' or func == 'Zero_Loss_Peak'):
            model.set_parameters_value('area',
                                       func_dict[func][2],
                                       component_list = [func]
                                      )

            model.set_parameters_value('gamma',
                                       func_dict[func][1] / 2,
                                       component_list = [func]
                                      )

            model.set_parameters_value('FWHM',
                                       func_dict[func][1] / 2,
                                       component_list = [func]
                                      )

            if (func != 'Second_Plasmon_Peak'):
                model.set_parameters_value('centre',
                                           func_dict[func][0],
                                           component_list = [func]
                                          )

        if (self.function_set == 'Lorentzian' and func != 'Zero_Loss_Peak'):
            model.set_parameters_value('A',
                                       func_dict[func][2],
                                       component_list = [func]
                                      )

            model.set_parameters_value('gamma',
                                       func_dict[func][1] / 2,
                                       component_list = [func]
                                      )

            if (func != 'Second_Plasmon_Peak'):
                model.set_parameters_value('centre',
                                           func_dict[func][0],
                                           component_list = [func]
                                          )

        if (self.function_set == 'Gaussian' and func != 'Zero_Loss_Peak'):
            model.set_parameters_value('A',
                                       func_dict[func][2],
                                       component_list = [func]
                                      )

            model.set_parameters_value('sigma',
                                       func_dict[func][1] / (2 * np.sqrt( np.log(2) )),
                                       component_list = [func]
                                      )

            if (func != 'Second_Plasmon_Peak'):
                model.set_parameters_value('centre',
                                           func_dict[func][0],
                                           component_list = [func]
                                          )

        if (self.function_set == 'VolumePlasmonDrude' and func != 'Zero_Loss_Peak'):
            model.set_parameters_value('intensity',
                                       func_dict[func][2],
                                       component_list = [func]
                                      )

            model.set_parameters_value('fwhm',
                                       func_dict[func][1],
                                       component_list = [func]
                                      )

            if (func != 'Second_Plasmon_Peak'):
                model.set_parameters_value('plasmon_energy',
                                           func_dict[func][0],
                                           component_list = [func]
                                          )
    
    
    def set_by_mean(self,
                    model,
                    mean,
                    func
                   ):
        """
        Initializing starting parameters by estimation of mean spectrum of the spectrum image
        """
        func_dict = {'Zero_Loss_Peak'      : mean.components.Zero_Loss_Peak,
                     'First_Plasmon_Peak'  : mean.components.First_Plasmon_Peak,
                     'Second_Plasmon_Peak' : mean.components.Second_Plasmon_Peak
                    }
        
        if (self.function_set == 'Voigt' or func == 'Zero_Loss_Peak'):
            model.set_parameters_value('area',
                                       func_dict[func].area.value,
                                       component_list = [func]
                                      )

            model.set_parameters_value('gamma',
                                       func_dict[func].gamma.value,
                                       component_list = [func]
                                      )

            model.set_parameters_value('FWHM',
                                       func_dict[func].FWHM.value,
                                       component_list = [func]
                                      )

            if (func != 'Second_Plasmon_Peak'):
                model.set_parameters_value('centre',
                                           func_dict[func].centre.value,
                                           component_list = [func]
                                          )

        if (self.function_set == 'Lorentzian' and func != 'Zero_Loss_Peak'):
            model.set_parameters_value('A',
                                       func_dict[func].A.value,
                                       component_list = [func]
                                      )

            model.set_parameters_value('gamma',
                                       func_dict[func].gamma.value,
                                       component_list = [func]
                                      )

            if (func != 'Second_Plasmon_Peak'):
                model.set_parameters_value('centre',
                                           func_dict[func].centre.value,
                                           component_list = [func]
                                          )

        if (self.function_set == 'Gaussian' and func != 'Zero_Loss_Peak'):
            model.set_parameters_value('A',
                                       func_dict[func].A.value,
                                       component_list = [func]
                                      )

            model.set_parameters_value('sigma',
                                       func_dict[func].sigma.value,
                                       component_list = [func]
                                      )

            if (func != 'Second_Plasmon_Peak'):
                model.set_parameters_value('centre',
                                           func_dict[func].centre.value,
                                           component_list = [func]
                                          )

        if (self.function_set == 'VolumePlasmonDrude' and func != 'Zero_Loss_Peak'):
            model.set_parameters_value('intensity',
                                       func_dict[func].intensity.value,
                                       component_list = [func]
                                      )

            model.set_parameters_value('fwhm',
                                       func_dict[func].fwhm.value,
                                       component_list = [func]
                                      )

            if (func != 'Second_Plasmon_Peak'):
                model.set_parameters_value('plasmon_energy',
                                           func_dict[func].plasmon_energy.value,
                                           component_list = [func]
                                          )
    
            
    def set_model_params(self, 
                         model, 
                         mean   = None,
                         params = None,
                         func='Zero_Loss_Peak'
                        ):
        """
        Comments missing
        """
        if (params != None):
            self.set_by_params(model, params, func)
            
        elif (mean != None):
            self.set_by_mean(model, mean, func)
    
    
    def fit_eels(self, 
                 fitter, 
                 method,
                 auto,
                 lazy
                ):
        """
        Serial model fitting calculation
        
            fitter: the optimizer chosen - see hyperspy documentation for 
                    supported optimizers

            method: the minimization method (least squared - 'ls'
                                             maximum likelyhood - 'ml'
                                             optional: 'custom' - support for 
                                                       self written minimization method
                                            )

            auto:   If True: possible adjustments to the starting parameters are possible
                             before start of fitting routine accessible by gui

            lazy:   usage of dask-arrays instead of standard numpy-arrays to minimize
                    memory demand for large datasets (> 4GB | e.g. '.dm4' - files)
        """
        if (self.attr_deconv == False):
            print('Reinitialising poissonian noise estimation...')
            self.File.estimate_poissonian_noise_variance()
            
            if (lazy == True):
                self.File.metadata.Signal.Noise_properties.variance.compute()
            
            model, params = self.init_model()

            if (method == 'ls'):
                bounded = True

            else:
                bounded = False

            self.set_bounds(model, 
                            bounded
                           )

            offset = self.File.axes_manager['Energy loss'].offset
            scale  = self.File.axes_manager['Energy loss'].scale
            size   = self.File.axes_manager['Energy loss'].size

            e_max  = float(int(scale * size + offset))

            upper_bound = e_max

            self.set_second_plasmonenergy(model)
            
            self.fit_zlp_only(model, 
                              fitter, 
                              method,
                              bounded
                             )
            
            plasmonenergy = self.fit_pp_only(upper_bound,
                                             model, 
                                             fitter, 
                                             method,
                                             bounded
                                            )

            model.components.Zero_Loss_Peak.active      = True
            model.components.First_Plasmon_Peak.active  = True
            model.components.Second_Plasmon_Peak.active = True
            
            model.set_signal_range(-self.elastic_threshold, 
                                   2.5 * plasmonenergy
                                  )
            
            if (auto == False):
                gui = self.yes_or_no('Do you want to adjust the starting parameters?')
                if (gui == True):
                    self.model_gui(model)
            
            print('Correcting poissonian noise for the gain factor of\n' +
                  'the EELS - detector by generating statistics on the\n' +
                  'reduced Chi-squared.'
                 )
            
            mean_rchisq = self.poisson_noise_gain_correction(model,
                                                             fitter,
                                                             method,
                                                             bounded,
                                                             plasmonenergy
                                                            )
            
            model.multifit(fitter  = fitter, 
                           method  = method,
                           bounded = bounded
                          )
        
        elif (self.attr_deconv == True):
            print('Reinitialising poissonian noise estimation...')
            self.File_deconv.estimate_poissonian_noise_variance()
            
            if (lazy == True):
                self.File_deconv.metadata.Signal.Noise_properties.variance.compute()
                
            model, params = self.init_model()

            if (method == 'ls'):
                bounded = True

            else:
                bounded = False

            self.set_bounds(model, 
                            bounded
                           )

            offset = self.File_deconv.axes_manager['Energy loss'].offset
            scale  = self.File_deconv.axes_manager['Energy loss'].scale
            size   = self.File_deconv.axes_manager['Energy loss'].size

            e_max  = float(int(scale * size + offset))

            upper_bound = e_max

            model.components.Zero_Loss_Peak.active      = True
            model.components.First_Plasmon_Peak.active  = True
            model.components.Second_Plasmon_Peak.active = False
            
            plasmonenergy = self.fit_fpp_only(upper_bound,
                                              model, 
                                              fitter, 
                                              method,
                                              bounded
                                             )
            
            model.set_signal_range(self.elastic_threshold, 
                                   1.5 * plasmonenergy
                                  )
            
            if (auto == False):
                gui = self.yes_or_no('Do you want to adjust the starting parameters?')
                if (gui == True):
                    self.model_gui(model)
                    
            print('Correcting poissonian noise for the gain factor of\n' +
                  'the EELS - detector by generating statistics on the\n:' +
                  'reduced Chi-squared.'
                 )
            
            mean_rchisq = self.poisson_noise_gain_correction(model,
                                                             fitter,
                                                             method,
                                                             bounded,
                                                             plasmonenergy
                                                            )
            
            model.multifit(fitter  = fitter, 
                           method  = method,
                           bounded = bounded
                          )
        
        self.Chisq       = model.chisq
        self.red_Chisq   = model.red_chisq
        self.rchisq_mean = np.mean(
            self.red_Chisq.data[np.invert(np.isnan(self.red_Chisq.data))]
        )
        self.rchisq_std  = np.std(
            self.red_Chisq.data[np.invert(np.isnan(self.red_Chisq.data))]
        )

        print('Adj. fit goodness (reduced Chi squared): ', self.rchisq_mean)
        print('Standard deviation of adj. fit goodness: ', self.rchisq_std)
        
        return model
    
    
    def fit_eels_SAMF(self, 
                      fitter, 
                      method, 
                      multithreading, 
                      workers,
                      auto,
                      lazy
                     ):
        """
        SAMFire model fitting calculation 
        
            fitter: the optimizer chosen - see hyperspy documentation for 
                    supported optimizers

            method: the minimization method (least squared - 'ls'
                                             maximum likelyhood - 'ml'
                                             optional: 'custom' - support for 
                                                       self written minimization method
                                            )

            auto:   If True: possible adjustments to the starting parameters are possible
                             before start of fitting routine accessible by gui

            lazy:   usage of dask-arrays instead of standard numpy-arrays to minimize
                    memory demand for large datasets (> 4GB | e.g. '.dm4' - files)
        """
        if (self.attr_deconv == False):
            print('Reinitialising poissonian noise estimation...')
            self.File.estimate_poissonian_noise_variance()
            
            if (lazy == True):
                self.File.metadata.Signal.Noise_properties.variance.compute()
                
            model, params = self.init_model()

            if (method == 'ls'):
                bounded = True

            else:
                bounded = False

            self.set_bounds(model, bounded)

            offset = self.File.axes_manager['Energy loss'].offset
            scale  = self.File.axes_manager['Energy loss'].scale
            size   = self.File.axes_manager['Energy loss'].size

            e_max  = float(int(scale * size + offset))

            upper_bound = e_max
        
            self.set_second_plasmonenergy(model)
            
            self.fit_zlp_only(model, 
                              fitter, 
                              method,
                              bounded
                             )
            
            self.fit_pp_only(upper_bound,
                             model, 
                             fitter, 
                             method,
                             bounded
                            )
            
            if (self.function_set == 'VolumePlasmonDrude'):
                plasmonenergy = np.nanmean(
                    model.components.First_Plasmon_Peak.plasmon_energy.as_signal(
                        field='values'
                    ).data
                )
                
            
            else:
                plasmonenergy = np.nanmean(
                    model.components.First_Plasmon_Peak.centre.as_signal(
                        field='values'
                    ).data
                )
            
            model.components.Zero_Loss_Peak.active      = True
            model.components.First_Plasmon_Peak.active  = True
            model.components.Second_Plasmon_Peak.active = True
            
            model.set_signal_range(-self.elastic_threshold, 
                                   2.5 * plasmonenergy
                                  )
            if (auto == False):
                gui = self.yes_or_no('Do you want to adjust the starting parameters?')
                if (gui == True):
                    self.model_gui(model)
            
            print('Correcting poissonian noise for the gain factor of\n' +
                  'the EELS - detector by generating statistics on the\n:' +
                  'reduced Chi-squared.'
                 )
            
            mean_rchisq = self.poisson_noise_gain_correction(model,
                                                             fitter,
                                                             method,
                                                             bounded,
                                                             plasmonenergy
                                                            )
            
            print('Generating_seeds for SAMFire...')
            
            self.generate_seeds(model,
                                fitter,
                                method,
                                bounded,
                                plasmonenergy
                               )
            
            samf = model.create_samfire(workers=workers, 
                                        ipyparallel=multithreading, 
                                        setup=True
                                       )

            samf.metadata.goodness_test.tolerance = mean_rchisq * 1.5
            samf.remove(1)
            samf.refresh_database()

            samf.start(fitter=fitter, 
                       method=method,
                       bounded=bounded
                      )

            plt.close()
            
        
        elif (self.attr_deconv == True):
            print('Reinitialising poissonian noise estimation...')
            self.File_deconv.estimate_poissonian_noise_variance()

            if (lazy == True):
                self.File_deconv.metadata.Signal.Noise_properties.variance.compute()
                
            model, params = self.init_model()

            if (method == 'ls'):
                bounded = True

            else:
                bounded = False

            self.set_bounds(model, bounded)

            offset = self.File.axes_manager['Energy loss'].offset
            scale  = self.File.axes_manager['Energy loss'].scale
            size   = self.File.axes_manager['Energy loss'].size

            e_max  = float(int(scale * size + offset))

            upper_bound = e_max

            model.components.Zero_Loss_Peak.active      = True
            model.components.First_Plasmon_Peak.active  = True
            model.components.Second_Plasmon_Peak.active = False
            
            plasmonenergy = self.fit_fpp_only(upper_bound,
                                              model, 
                                              fitter,
                                              method,
                                              bounded
                                             )
            
            model.set_signal_range(self.elastic_threshold, 
                                   1.5 * plasmonenergy
                                  )
            
            if (auto == False):
                gui = self.yes_or_no('Do you want to adjust the starting parameters?')
                if (gui == True):
                    self.model_gui(model)
            
            print('Correcting poissonian noise for the gain factor of\n' +
                  'the EELS - detector by generating statistics on the\n' +
                  'reduced Chi-squared.'
                 )
            
            mean_rchisq = self.poisson_noise_gain_correction(model,
                                                             fitter,
                                                             method,
                                                             bounded,
                                                             plasmonenergy
                                                            )
            
            print('Generating_seeds for SAMFire...')
            
            self.generate_seeds(model,
                                fitter,
                                method,
                                bounded,
                                plasmonenergy
                               )
            
            samf = model.create_samfire(workers=workers, 
                                        ipyparallel=multithreading, 
                                        setup=True
                                       )

            samf.metadata.goodness_test.tolerance = mean_rchisq * 1.5
            samf.remove(1)
            samf.refresh_database()

            samf.start(fitter=fitter, 
                       method=method,
                       bounded=bounded
                      )
        
        try:
            self.Chisq       = model.chisq
            self.red_Chisq   = model.red_chisq
            self.rchisq_mean = np.mean(
                self.red_Chisq.data[np.invert(np.isnan(self.red_Chisq.data))]
            )
            self.rchisq_std  = np.std(
                self.red_Chisq.data[np.invert(np.isnan(self.red_Chisq.data))]
            )

            print('Adj. fit goodness (reduced Chi squared): ', self.rchisq_mean)
            print('Standard deviation of adj. fit goodness: ', self.rchisq_std)
        except:
            print('\nAttention: Reduced Chi squared calculation failed. ')
        return model
    
    
    def generate_seeds(self,
                       model,
                       fitter,
                       method,
                       bounded,
                       plasmonenergy
                      ):
        """
        Calling serial fit routine for a #number of pixels to gather statistics
        on the reduced chi squared.
        (#number = steps|steps^2, dependence of navigation dimension) 
        
        The pixel selection is homogenously distributed over the navigation space,
        to estimate local deviations of the mean parameter solutions.
        This way, SAMFire will have optimally close solutions for the parameter
        space to estimate neighbouring pixels and propagate fast.
        
        It is a requirement for SAMFire.
        For multifit routine: functionality is only needed for gain correction!
        """
        steps = 10
        
        if (self.attr_deconv == True):
            model.components.Zero_Loss_Peak.active      = True
            model.components.First_Plasmon_Peak.active  = True
            model.components.Second_Plasmon_Peak.active = False
            model.set_signal_range(self.elastic_threshold, 
                                   1.5 * plasmonenergy
                                  )
        
        else:
            model.components.Zero_Loss_Peak.active      = True
            model.components.First_Plasmon_Peak.active  = True
            model.components.Second_Plasmon_Peak.active = True 
            model.set_signal_range(-self.elastic_threshold, 
                                   2.5 * plasmonenergy
                                  )
        
        if (model.axes_manager.navigation_dimension == 1):
            
            for i in range(steps):
                step_x = int(self.File.axes_manager.navigation_shape[0]/steps)

                model.axes_manager.indices = (int(step_x/2+i*step_x),)

                model.fit(fitter=fitter, 
                          method=method,
                          bounded=bounded
                         )


        elif (model.axes_manager.navigation_dimension == 2):

            for i in range(steps):
                for j in range(steps):
                    step_x = int(self.File.axes_manager.navigation_shape[0]/steps)
                    step_y = int(self.File.axes_manager.navigation_shape[1]/steps)

                    model.axes_manager.indices = (int(step_x/2 + i*step_x), 
                                                  int(step_y/2 + j*step_y))
                    model.fit(fitter=fitter, 
                              method=method,
                              bounded=bounded
                             )
                        
        
    def poisson_noise_gain_correction(self,
                                      model,
                                      fitter,
                                      method,
                                      bounded,
                                      plasmonenergy
                                     ):
        """
        Adjusting the poissonian noise by considering the influence of the 
        gain factor, as it is a multiplier attached to the poissonian noise.
        
        This will use the reduced chi squared to estimate the gain factor by
        assuming a optimal model, setting the mean chi squared to 1.
        
        Therefore all information on the fit goodness of the model is adjusted.
        """
        if (self.attr_deconv == True):
            model.components.Zero_Loss_Peak.active      = True
            model.components.First_Plasmon_Peak.active  = True
            model.components.Second_Plasmon_Peak.active = False
            model.set_signal_range(self.elastic_threshold, 
                                   1.5 * plasmonenergy
                                  )
        
        else:
            model.components.Zero_Loss_Peak.active      = True
            model.components.First_Plasmon_Peak.active  = True
            model.components.Second_Plasmon_Peak.active = True 
            model.set_signal_range(-self.elastic_threshold, 
                                   2.5 * plasmonenergy
                                  )
        
        self.generate_seeds(model,
                            fitter,
                            method,
                            bounded,
                            plasmonenergy
                           )

        red_chisq = model.red_chisq.data
        mean_rchisq = np.mean(red_chisq[np.invert(np.isnan(red_chisq))])
        
        if (self.attr_deconv == False):
            print('Including gain factor of detector in poissonian noise. Rescaling...')
            self.File.estimate_poissonian_noise_variance(gain_factor=mean_rchisq)
            
            if (self.is_lazy == True):
                self.File.metadata.Signal.Noise_properties.variance.compute()
        
        else:
            print('Including gain factor of detector in poissonian noise. Rescaling...')
            self.File_deconv.estimate_poissonian_noise_variance(gain_factor=mean_rchisq)
            
            if (self.is_lazy == True):
                self.File_deconv.metadata.Signal.Noise_properties.variance.compute()
            
        return mean_rchisq
    
    
    def save_models_to_file(self, 
                            filename=''
                           ):
        """
        Saving all the currently calculated models for the working file to a new file
        #filename#
        """
        if (filename == ''):
            filename = asksaveasfile(mode='w', defaultextension=".hspy")
        if (self.attr_deconv == False):
            self.File.save(filename)
        
        else:
            self.File_deconv.save(filename)

    
    def load_model(self, mkey=None):
        """
        Loading a model picked by the user from file by firstly recovering all models
        and then loading the picked #model_name# to the class variable #Fit_Model#
        """
        if (self.is_lazy == True):
            try:
                if (self.attr_deconv == False):
                    self.File.compute()

                else:
                    self.File_deconv.compute()
                    
            except:
                pass
                
        available = np.empty((len(self.all_models)), 
                             dtype=bool
                            )
        
        available = self.restore_models_to_dict(available)
        print('Available models: \n')
        for key in self.all_models:
            if (self.all_models[key] in self.models_dict):
                print('"' + 
                      str(key) + 
                      '" : ', 
                      self.all_models[key], 
                      '\n'
                     )
        
        if (available[available].size == 0):
            print('No model is available. Exiting.')
            return None
        
        if (mkey == None):
            print('\nIf you want to exit the model loading process,' + 
                  'please type: ("exit"/"cancel")'
                 )
            mkey = input("Which model should be loaded? ")

            if (mkey == 'exit' or mkey == 'cancel'):
                print('Loading process is interrupted by user input.')
                if (self.is_lazy == True):
                    if (self.attr_deconv == False):
                        self.File = self.File.as_lazy()

                    else:
                        self.File_deconv = self.File_deconv.as_lazy()
                return None
        
        
            elif (mkey in self.all_models.keys() and available[int(mkey)] == True):
                print('Loading parameter maps for: ' + 
                      str(self.all_models[mkey])
                     )
                self.model_name   = self.all_models[mkey]

                self.get_model()
                name = self.split(self.all_models[mkey], ('_'))
                
                self.function_set = name[0]
                self.optimizer    = name[1]
                self.method       = name[2]
                self.generate_param_maps(self.method)
                self.Chisq        = self.Fit_Model.chisq
                self.red_Chisq    = self.Fit_Model.red_chisq

            else:
                try:
                    print('Loading parameter maps for: ' + str(mkey))
                    self.model_name   = mkey

                    self.get_model()
                    name = self.split(self.model_name, ('_'))

                    self.function_set = name[0]
                    self.optimizer    = name[1]
                    self.method       = name[2]
                    self.generate_param_maps(self.method)
                    
                    self.Chisq        = self.Fit_Model.chisq
                    self.red_Chisq    = self.Fit_Model.red_chisq

                except:
                    print('Your input did not match with any existing model of the' +
                          'loaded file, please try again.')
                    self.load_model()

            method = self.split(self.all_models[mkey], ('_'))[-1]
            self.generate_param_maps(method)
        
        else:
            if (str(mkey) in self.all_models.keys() and available[int(mkey)] == True):
                print('Loading parameter maps for: ' + 
                      str(self.all_models[str(mkey)])
                     )
                self.model_name   = self.all_models[str(mkey)]

                self.get_model()
                name = self.split(self.all_models[mkey], ('_'))
                
                self.function_set = name[0]
                self.optimizer    = name[1]
                self.method       = name[2]
                self.generate_param_maps(self.method)
                
                self.function_set = self.split(self.all_models[str(mkey)], ('_'))[0]
                self.Chisq        = self.Fit_Model.chisq
                self.red_Chisq    = self.Fit_Model.red_chisq
            
            else:
                try:
                    print('Loading parameter maps for: ' + str(mkey))
                    self.model_name   = mkey

                    self.get_model()
                    name = self.split(self.model_name, ('_'))

                    self.function_set = name[0]
                    self.optimizer    = name[1]
                    self.method       = name[2]
                    self.generate_param_maps(self.method)
                    
                    self.Chisq        = self.Fit_Model.chisq
                    self.red_Chisq    = self.Fit_Model.red_chisq

                except:
                    print('Your input did not match with any existing model of the' +
                          'loaded file, please try again.')
                    self.load_model()
        
        if (self.is_lazy == True):
            if (self.attr_deconv == False):
                self.File = self.File.as_lazy()

            else:
                self.File_deconv = self.File_deconv.as_lazy()
        
        print('Finished loading process.')
            
        
    def restore_models_to_dict(self,
                               available
                              ):
        """
        Adding all accessible models of loaded file to the model dictionary #models_dict#
        """
        for key in range(len(available)):
            
            try:
                if (self.all_models[str(key)] in self.models_dict):
                    available[key] = True
                    
                elif (self.attr_deconv == False):
                    self.model_name                   = self.all_models[str(key)]
                    self.models_dict[self.model_name] = self.File.models.restore(
                        self.model_name
                    )
                    available[key] = True
                    
                else:
                    self.model_name                   = self.all_models[str(key)]
                    self.models_dict[self.model_name] = self.File_deconv.models.restore(
                        self.model_name
                    )
                    available[key] = True
                    
            except:
                available[key] = False
        
        return available
    
    
    def update_models_dict(self):
        """
        Updating #models_dict# by adding the currently picked model chosen by #model_name#
        """
        if (self.attr_deconv == False):
            self.models_dict[self.model_name] = self.File.models.restore(
                self.model_name
            )
            
        else:
            self.models_dict[self.model_name] = self.File_deconv.models.restore(
                self.model_name
            )
        
    
    def get_model(self):
        """
        Add the selected model to the class variable #Fit_Model#
        """
        self.Fit_Model  = self.models_dict[self.model_name]
    
    
    #####################################################################################
    #   The following functions are used to transform parameters, calculate E_p(q=0)    #
    # and calculate the gaussian error propagation for each corresponding parameter map #
    #####################################################################################
    
    def voigt_fwhm_gauss_propagation(self,
                                     func
                                    ):
        func_dict = {'Zero_Loss_Peak'      : self.Fit_Model.components.Zero_Loss_Peak,
                     'First_Plasmon_Peak'  : self.Fit_Model.components.First_Plasmon_Peak,
                     'Second_Plasmon_Peak' : self.Fit_Model.components.Second_Plasmon_Peak
                    }

        gamma_zlp = func_dict[func].gamma.as_signal(
            field = 'values'
        )

        L_fwhm    = np.add(gamma_zlp, 
                           gamma_zlp
                          )

        G_fwhm    = func_dict[func].FWHM.as_signal(
            field = 'values'
        )

        gamma_zlp = func_dict[func].gamma.as_signal(
            field = 'std'
        )

        L_fwhm_std = np.add(gamma_zlp, 
                            gamma_zlp
                           )

        G_fwhm_std = func_dict[func].FWHM.as_signal(
            field = 'std'
        )

        dfwhm_dl = (L_fwhm * 0.2166 / ( L_fwhm**2 * 0.2166 + 
                                       G_fwhm**2 )**(1/2) + 
                    0.5346
                   )

        dfwhm_dg = (G_fwhm / ( L_fwhm ** 2 * 0.2166 + 
                              G_fwhm ** 2 )**(1/2)          
                   )

        error = np.sqrt((G_fwhm_std * dfwhm_dg) ** 2 + 
                        (L_fwhm_std * dfwhm_dl) ** 2 
                       )

        return error
    
    
    def Ep_q0_gauss_propagation(self):
        emax  = self.FPP_Emax 
        fwhm  = self.FPP_FWHM
        
        Ep_q0 = ( emax ** 2 - (fwhm / 2) ** 2 ) ** 0.5
        
        dEp_demax = emax * 2 / ( emax ** 2 * 4 + 
                                fwhm ** 2 
                               ) ** (1/2)
        dEp_dfwhm = fwhm * (1/2) / ( emax ** 2 * 4 + 
                                  fwhm ** 2 
                                 ) ** (1/2)
        
        emax_std = self.FPP_Emax.metadata.Signal.Noise_properties.variance ** (1/2)
        fwhm_std = self.FPP_FWHM.metadata.Signal.Noise_properties.variance ** (1/2)
        
        Ep_q0_std = ( ( dEp_demax * emax_std ) ** 2 + ( dEp_dfwhm * fwhm_std ) ** 2 ) ** (1/2)
        
        return Ep_q0_std
        
    
    def param_maps_drude(self):
        self.FPP_FWHM   = self.Fit_Model.components.First_Plasmon_Peak.fwhm.as_signal(
            field = 'values'
        )
        
        self.FPP_Emax   = self.Fit_Model.components.First_Plasmon_Peak.plasmon_energy.as_signal(
            field = 'values'
        )
        
        self.FPP_Int    = self.Fit_Model.components.First_Plasmon_Peak.intensity.as_signal(
            field = 'values'
        )

        if (self.attr_deconv == False):
            gamma_zlp = self.Fit_Model.components.Zero_Loss_Peak.gamma.as_signal(
                field = 'values'
            )
            
            ZLP_L_fwhm      = np.add(gamma_zlp, 
                                     gamma_zlp
                                    )
            
            ZLP_G_fwhm      = self.Fit_Model.components.Zero_Loss_Peak.FWHM.as_signal(
                field = 'values'
            )
            
            self.ZLP_FWHM   = (ZLP_L_fwhm*0.5346 + 
                               np.sqrt(ZLP_G_fwhm**2 + ZLP_L_fwhm**2*0.2166)
                              )
            
            self.ZLP_Emax   = self.Fit_Model.components.Zero_Loss_Peak.centre.as_signal(
                field = 'values'
            )
            
            self.ZLP_Int    = self.Fit_Model.components.Zero_Loss_Peak.area.as_signal(
                field = 'values'
            )
            

            self.SPP_FWHM   = self.Fit_Model.components.Second_Plasmon_Peak.fwhm.as_signal(
                field = 'values'
            )
            
            self.SPP_Emax   = self.Fit_Model.components.Second_Plasmon_Peak.plasmon_energy.as_signal(
                field = 'values'
            )
            
            self.SPP_Int    = self.Fit_Model.components.Second_Plasmon_Peak.intensity.as_signal(
                field = 'values'
            )

        self.Ep_q0      = ( self.FPP_Emax**2 - (self.FPP_FWHM / 2)**2 )**0.5
        
    
    def std_maps_drude(self):
        self.FPP_FWHM.metadata.Signal.set_item(
            "Noise_properties.variance", 
            self.Fit_Model.components.First_Plasmon_Peak.fwhm.as_signal(
                field = 'std'
            ) ** 2
        )
        
        self.FPP_Emax.metadata.Signal.set_item(
            "Noise_properties.variance", 
            self.Fit_Model.components.First_Plasmon_Peak.plasmon_energy.as_signal(
                field = 'std'
            ) ** 2
        )
        
        self.FPP_Int.metadata.Signal.set_item(
            "Noise_properties.variance", 
            self.Fit_Model.components.First_Plasmon_Peak.intensity.as_signal(
                field = 'std'
            ) ** 2
        )

        if (self.attr_deconv == False):
            error = self.voigt_fwhm_gauss_propagation('Zero_Loss_Peak')
            
            self.ZLP_FWHM.metadata.Signal.set_item(
                "Noise_properties.variance", 
                error ** 2
            )
            
            self.ZLP_Emax.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Zero_Loss_Peak.centre.as_signal(
                    field = 'std'
                ) ** 2
            )
            
            self.ZLP_Int.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Zero_Loss_Peak.area.as_signal(
                    field = 'std'
                ) ** 2
            )

            self.SPP_FWHM.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Second_Plasmon_Peak.fwhm.as_signal(
                    field = 'std'
                ) ** 2
            )
            
            self.SPP_Emax.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Second_Plasmon_Peak.plasmon_energy.as_signal(
                    field = 'std'
                ) ** 2
            )
            
            self.SPP_Int.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Second_Plasmon_Peak.intensity.as_signal(
                    field = 'std'
                ) ** 2
            )
        

        Ep_q0_std = self.Ep_q0_gauss_propagation()
        
        self.Ep_q0.metadata.Signal.set_item(
            "Noise_properties.variance", 
            Ep_q0_std ** 2
        )
        
    
    def param_maps_lorentzian(self):
        gamma_fpp = self.Fit_Model.components.First_Plasmon_Peak.gamma.as_signal(
            field = 'values'
        )
        
        self.FPP_FWHM   = np.add(gamma_fpp, gamma_fpp)
        
        self.FPP_Emax   = self.Fit_Model.components.First_Plasmon_Peak.centre.as_signal(
            field = 'values'
        )
        
        self.FPP_Int    = self.Fit_Model.components.First_Plasmon_Peak.A.as_signal(
            field = 'values'
        )

        if (self.attr_deconv == False):
            gamma_zlp = self.Fit_Model.components.Zero_Loss_Peak.gamma.as_signal(
                field = 'values'
            )
            
            ZLP_L_fwhm      = np.add(gamma_zlp, 
                                     gamma_zlp
                                    )
            
            ZLP_G_fwhm      = self.Fit_Model.components.Zero_Loss_Peak.FWHM.as_signal(
                field = 'values'
            )
            
            self.ZLP_FWHM   = (ZLP_L_fwhm*0.5346 + 
                               np.sqrt(ZLP_G_fwhm**2 + 
                                       ZLP_L_fwhm**2*0.2166
                                      )
                              )
            
            self.ZLP_Emax   = self.Fit_Model.components.Zero_Loss_Peak.centre.as_signal(
                field = 'values'
            )
            
            self.ZLP_Int    = self.Fit_Model.components.Zero_Loss_Peak.area.as_signal(
                field = 'values'
            )
            

            gamma_spp = self.Fit_Model.components.Second_Plasmon_Peak.gamma.as_signal(
                field = 'values'
            )
            
            self.SPP_FWHM   = np.add(gamma_spp, 
                                     gamma_spp
                                    )
            
            self.SPP_Emax   = self.Fit_Model.components.Second_Plasmon_Peak.centre.as_signal(
                field = 'values'
            )
            
            self.SPP_Int    = self.Fit_Model.components.Second_Plasmon_Peak.A.as_signal(
                field = 'values'
            )
        

        self.Ep_q0      = ( self.FPP_Emax**2 - (self.FPP_FWHM / 2)**2 )**0.5
    
    
    def std_maps_lorentzian(self):
        self.FPP_FWHM.metadata.Signal.set_item(
            "Noise_properties.variance", 
            self.Fit_Model.components.First_Plasmon_Peak.gamma.as_signal(
                field = 'std'
            ) ** 2
        )
        
        self.FPP_Emax.metadata.Signal.set_item(
            "Noise_properties.variance", 
            self.Fit_Model.components.First_Plasmon_Peak.centre.as_signal(
                field = 'std'
            ) ** 2
        )
        
        self.FPP_Int.metadata.Signal.set_item(
            "Noise_properties.variance", 
            self.Fit_Model.components.First_Plasmon_Peak.A.as_signal(
                field = 'std'
            ) ** 2
        )

        if (self.attr_deconv == False):
            error = self.voigt_fwhm_gauss_propagation('Zero_Loss_Peak')
            
            self.ZLP_FWHM.metadata.Signal.set_item(
                "Noise_properties.variance", 
                error ** 2
            )
            
            self.ZLP_Emax.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Zero_Loss_Peak.centre.as_signal(
                    field = 'std'
                ) ** 2
            )
            
            self.ZLP_Int.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Zero_Loss_Peak.area.as_signal(
                    field = 'std'
                ) ** 2
            )
            
            
            self.SPP_FWHM.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Second_Plasmon_Peak.gamma.as_signal(
                    field = 'std'
                ) ** 2
            )
            
            self.SPP_Emax.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Second_Plasmon_Peak.centre.as_signal(
                    field = 'std'
                ) ** 2
            )
            
            self.SPP_Int.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Second_Plasmon_Peak.A.as_signal(
                    field = 'std'
                ) ** 2
            )
        

        Ep_q0_std = self.Ep_q0_gauss_propagation()
        
        self.Ep_q0.metadata.Signal.set_item(
            "Noise_properties.variance", 
            Ep_q0_std ** 2
        )
        
    
    def param_maps_gaussian(self):
        self.FPP_FWHM   = self.Fit_Model.components.First_Plasmon_Peak.sigma.as_signal(
                field = 'values'
        )*2*np.sqrt(np.log(2))
        
        self.FPP_Emax   = self.Fit_Model.components.First_Plasmon_Peak.centre.as_signal(
            field = 'values'
        )
        
        self.FPP_Int    = self.Fit_Model.components.First_Plasmon_Peak.A.as_signal(
            field = 'values'
        )

        if (self.attr_deconv == False):
            gamma_zlp = self.Fit_Model.components.Zero_Loss_Peak.gamma.as_signal(
                field = 'values'
            )
            
            ZLP_L_fwhm      = np.add(gamma_zlp, 
                                     gamma_zlp
                                    )
            
            ZLP_G_fwhm      = self.Fit_Model.components.Zero_Loss_Peak.FWHM.as_signal(
                field = 'values'
            )
            
            self.ZLP_FWHM   = (ZLP_L_fwhm*0.5346 + 
                               np.sqrt(ZLP_G_fwhm**2 + ZLP_L_fwhm**2*0.2166)
                              )
            
            self.ZLP_Emax   = self.Fit_Model.components.Zero_Loss_Peak.centre.as_signal(
                field = 'values'
            )
            
            self.ZLP_Int    = self.Fit_Model.components.Zero_Loss_Peak.area.as_signal(
                field = 'values'
            )

            
            self.SPP_FWHM   = self.Fit_Model.components.Second_Plasmon_Peak.sigma.as_signal(
                field = 'values'
            )*2*np.sqrt(np.log(2))
            
            self.SPP_Emax   = self.Fit_Model.components.Second_Plasmon_Peak.centre.as_signal(
                field = 'values'
            )
            
            self.SPP_Int    = self.Fit_Model.components.Second_Plasmon_Peak.A.as_signal(
                field = 'values'
            )
        

        self.Ep_q0      = ( self.FPP_Emax**2 - (self.FPP_FWHM / 2)**2 )**0.5
        
    
    def std_maps_gaussian(self):
        self.FPP_FWHM.metadata.Signal.set_item(
            "Noise_properties.variance",
            self.Fit_Model.components.First_Plasmon_Peak.sigma.as_signal(
                field = 'std'
            ) ** 2
        )
        
        self.FPP_Emax.metadata.Signal.set_item(
            "Noise_properties.variance", 
            self.Fit_Model.components.First_Plasmon_Peak.centre.as_signal(
                field = 'std'
            ) ** 2
        )
        
        self.FPP_Int.metadata.Signal.set_item(
            "Noise_properties.variance", 
            self.Fit_Model.components.First_Plasmon_Peak.A.as_signal(
                field = 'std'
            ) ** 2
        )

        if (self.attr_deconv == False):
            error = self.voigt_fwhm_gauss_propagation('Zero_Loss_Peak')
            
            self.ZLP_FWHM.metadata.Signal.set_item(
                "Noise_properties.variance", 
                error ** 2
            )
            
            self.ZLP_Emax.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Zero_Loss_Peak.centre.as_signal(
                    field = 'std'
                ) ** 2
            )
            
            self.ZLP_Int.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Zero_Loss_Peak.area.as_signal(
                    field = 'std'
                ) ** 2
            )

            
            self.SPP_FWHM.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Second_Plasmon_Peak.sigma.as_signal(
                    field = 'std'
                ) ** 2
            )
            
            self.SPP_Emax.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Second_Plasmon_Peak.centre.as_signal(
                    field = 'std'
                ) ** 2
            )
            
            self.SPP_Int.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Second_Plasmon_Peak.A.as_signal(
                    field = 'std'
                ) ** 2
            )

        Ep_q0_std = self.Ep_q0_gauss_propagation()
        
        self.Ep_q0.metadata.Signal.set_item(
            "Noise_properties.variance", 
            Ep_q0_std ** 2
        )
    
    
    def param_maps_voigt(self):
        gamma_fpp = self.Fit_Model.components.First_Plasmon_Peak.gamma.as_signal(
                    field = 'values'
        )
        FPP_L_fwhm      = np.add(gamma_fpp, 
                                 gamma_fpp
                                )
        FPP_G_fwhm      = self.Fit_Model.components.First_Plasmon_Peak.FWHM.as_signal(
            field = 'values'
        )
        self.FPP_FWHM   = (FPP_L_fwhm*0.5346 + 
                           np.sqrt(FPP_G_fwhm**2 + 
                                   FPP_L_fwhm**2*0.2166
                                  )
                          )
        self.FPP_Emax   = self.Fit_Model.components.First_Plasmon_Peak.centre.as_signal(
            field = 'values'
        )
        self.FPP_Int    = self.Fit_Model.components.First_Plasmon_Peak.area.as_signal(
            field = 'values'
        )

        if (self.attr_deconv == False):
            gamma_zlp = self.Fit_Model.components.Zero_Loss_Peak.gamma.as_signal(
                field = 'values'
            )
            ZLP_L_fwhm      = np.add(gamma_zlp, 
                                     gamma_zlp
                                    )
            ZLP_G_fwhm      = self.Fit_Model.components.Zero_Loss_Peak.FWHM.as_signal(
                field = 'values'
            )
            self.ZLP_FWHM   = (ZLP_L_fwhm*0.5346 + 
                               np.sqrt(ZLP_G_fwhm**2 + 
                                       ZLP_L_fwhm**2*0.2166
                                      )
                              )
            self.ZLP_Emax   = self.Fit_Model.components.Zero_Loss_Peak.centre.as_signal(
                field = 'values'
            )
            self.ZLP_Int    = self.Fit_Model.components.Zero_Loss_Peak.area.as_signal(
                field = 'values'
            )

            gamma_spp = self.Fit_Model.components.Second_Plasmon_Peak.gamma.as_signal(
                field = 'values'
            )
            SPP_L_fwhm      = np.add(gamma_spp, 
                                     gamma_spp
                                    )
            SPP_G_fwhm      = self.Fit_Model.components.Second_Plasmon_Peak.FWHM.as_signal(
                field = 'values'
            )
            self.SPP_FWHM   = (SPP_L_fwhm*0.5346 + 
                               np.sqrt(SPP_G_fwhm**2 + 
                                       SPP_L_fwhm**2*0.2166
                                      )
                              )
            self.SPP_Emax   = self.Fit_Model.components.Second_Plasmon_Peak.centre.as_signal(
                field = 'values'
            )
            self.SPP_Int    = self.Fit_Model.components.Second_Plasmon_Peak.area.as_signal(
                field = 'values'
            )

        self.Ep_q0      = ( self.FPP_Emax**2 - (self.FPP_FWHM / 2)**2 )**0.5
        
    
    def std_maps_voigt(self):
        error = self.voigt_fwhm_gauss_propagation('First_Plasmon_Peak')
        
        self.FPP_FWHM.metadata.Signal.set_item(
            "Noise_properties.variance", 
            error ** 2
        )
        
        self.FPP_Emax.metadata.Signal.set_item(
            "Noise_properties.variance", 
            self.Fit_Model.components.First_Plasmon_Peak.centre.as_signal(
                field = 'std'
            ) ** 2
        )
        
        self.FPP_Int.metadata.Signal.set_item(
            "Noise_properties.variance", 
            self.Fit_Model.components.First_Plasmon_Peak.area.as_signal(
                field = 'std'
            ) ** 2
        )

        if (self.attr_deconv == False):
            error = self.voigt_fwhm_gauss_propagation('Zero_Loss_Peak')
            
            self.ZLP_FWHM.metadata.Signal.set_item(
                "Noise_properties.variance", 
                error ** 2
            )
            
            self.ZLP_Emax.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Zero_Loss_Peak.centre.as_signal(
                    field = 'std'
                ) ** 2
            )
            
            self.ZLP_Int.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Zero_Loss_Peak.area.as_signal(
                    field = 'std'
                ) ** 2
            )

            
            error = self.voigt_fwhm_gauss_propagation('Second_Plasmon_Peak')
            
            self.ZLP_FWHM.metadata.Signal.set_item(
                "Noise_properties.variance", 
                error ** 2
            )
            
            self.SPP_Emax.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Second_Plasmon_Peak.centre.as_signal(
                    field = 'std'
                ) ** 2
            )
            
            self.SPP_Int.metadata.Signal.set_item(
                "Noise_properties.variance", 
                self.Fit_Model.components.Second_Plasmon_Peak.area.as_signal(
                    field = 'std'
                ) ** 2
            )

        Ep_q0_std = self.Ep_q0_gauss_propagation()
        
        self.Ep_q0.metadata.Signal.set_item(
            "Noise_properties.variance", 
            Ep_q0_std ** 2
        )


    def generate_param_maps(self, 
                            method
                           ):
        """
        Generating parameter maps to estimate the plasmonic peakshift,
        which is calculated by [Egerton] as follows:  
                        
            $E_p (q=0) = \sqrt{ (E_max)^2 + (\frac{\hbar \Gamma}{2}) }$
        
        
         Standard deviation estimation of parameters by fitting is only
         supported by weighted least square method.
         It uses gaussian error propagation for some functional dependencies.
         E.g.: The uncertainty for $E_p (q=0)$ is calculated by gaussian
                error propagation (the standard deviations of the variables
                                   are propagated)
        """
        if (self.function_set == 'VolumePlasmonDrude'):
            self.param_maps_drude()
            
            if (method == 'ls'):
                self.std_maps_drude()
            
        elif (self.function_set == 'Lorentzian'):
            self.param_maps_lorentzian()
            if (method == 'ls'):
                self.std_maps_lorentzian()
        
        elif (self.function_set == 'Gaussian'):
            self.param_maps_gaussian()
            if (method == 'ls'):
                self.std_maps_gaussian()
            
        elif (self.function_set == 'Voigt'):
            self.param_maps_voigt()
            if (method == 'ls'):
                self.std_maps_voigt()
        
        else:
            print('No valid function set specified. Please look into docstring for further information.')
        
        print('Parameter images loaded. Setting properties...')
        
        units  = {r'Plasmon peak - $E_{\max}$'          : r'eV', 
                  r'Plasmon peak - $\Gamma$'            : r'eV', 
                  r'Plasmon peak - intensity'           : r'counts', 

                  r'Zero Loss peak - $E_{\max}$'        : r'eV', 
                  r'Zero Loss peak - $\Gamma$'          : r'eV',
                  r'Zero Loss peak - intensity'         : r'counts',

                  r'second Plasmon peak - $E_{\max}$'   : r'eV', 
                  r'second Plasmon peak - $\Gamma$'     : r'eV', 
                  r'second Plasmon peak - intensity'    : r'counts',

                  r'Plasmon energy - $E_{p}(q=0)$'      : r'eV', 
                  r'intensity ratio - $I_{pp}/I_{zlp}$' : r'perc.'
                 }
        if (self.attr_deconv == False):
            self.param_dict        = { r'Plasmon peak - $E_{\max}$'          : self.FPP_Emax,
                                       r'Plasmon peak - $\Gamma$'            : self.FPP_FWHM,
                                       r'Plasmon peak - intensity'           : self.FPP_Int,

                                       r'Zero Loss peak - $E_{\max}$'        : self.ZLP_Emax,
                                       r'Zero Loss peak - $\Gamma$'          : self.ZLP_FWHM,
                                       r'Zero Loss peak - intensity'         : self.ZLP_Int,

                                       r'second Plasmon peak - $E_{\max}$'   : self.SPP_Emax,
                                       r'second Plasmon peak - $\Gamma$'     : self.SPP_FWHM,
                                       r'second Plasmon peak - intensity'    : self.SPP_Int,

                                       r'Plasmon energy - $E_{p}(q=0)$'      : self.Ep_q0,
                                       r'intensity ratio - $I_{pp}/I_{zlp}$' : self.FPP_Int / self.ZLP_Int
                                     }
            for title in self.param_dict:
                self.param_dict[title].metadata.General.title = title
                self.param_dict[title].metadata.Signal.quantity = units[title]
            
        else:
            self.param_dict        = { r'Plasmon peak - $E_{\max}$'          : self.FPP_Emax,
                                       r'Plasmon peak - $\Gamma$'            : self.FPP_FWHM,
                                       r'Intensity - Plasmon peak'           : self.FPP_Int,

                                       r'Plasmon energy - $E_{p}(q=0)$'      : self.Ep_q0
                                     }
            for title in self.param_dict:
                self.param_dict[title].metadata.General.title = title
                self.param_dict[title].metadata.Signal.quantity = units[title]
    
        print('Finished loading parameter images.')
    
    
    def plot_file(self, 
                  darkfield = False,
                  slider    = False,
                  cmap      ='coolwarm'
                 ):
        """
        Plotting the spectrum image itself, seperating in a signal and 
        a navigation space where the navigation space corresponds to the
        probe spot.
        
            darkfield: If True - plotting the darkfield image as the
                                 navigation space instead of the total spectrum intensity
                                 
            slider:    If True - a slider is provided for the navigation
                                 space instead of the total spectrum intensity
        """
        if (self.attr_deconv == False):
            if (self.is_lazy == True):
                self.File.compute()

            title      = self.File.metadata.General.title
            title_corr = self.check_underscores_in_title(title)

            self.File.metadata.General.title = title_corr

            if (self.haadf != None and darkfield == True):
                self.File.plot(navigator = self.haadf, 
                               navigator_kwds=dict(colorbar=True, 
                                                   scalebar_color='black',
                                                   cmap=cmap,
                                                   axes_ticks=True
                                                  )
                              )

            elif (slider == True):
                self.File.plot(navigator = 'slider',
                               navigator_kwds=dict(colorbar=True, 
                                                   scalebar_color='black',
                                                   cmap=cmap,
                                                   axes_ticks=True
                                                  )
                              )

            else:
                self.File.plot(navigator='auto', 
                               navigator_kwds=dict(colorbar=True, 
                                                   scalebar_color='black',
                                                   cmap=cmap,
                                                   axes_ticks=True
                                                  )
                              )
        
        else:
            if (self.is_lazy == True):
                self.File_deconv.compute()

            title      = self.File_deconv.metadata.General.title
            title_corr = self.check_underscores_in_title(title)

            self.File_deconv.metadata.General.title = title_corr

            if (self.haadf != None and darkfield == True):
                self.File_deconv.plot(navigator = self.haadf, 
                                      navigator_kwds=dict(colorbar=True, 
                                                          scalebar_color='black',
                                                          cmap=cmap,
                                                          axes_ticks=True
                                                         )
                                     )

            elif (slider == True):
                self.File_deconv.plot(navigator = 'slider',
                                      navigator_kwds=dict(colorbar=True, 
                                                          scalebar_color='black',
                                                          cmap=cmap,
                                                          axes_ticks=True
                                                         )
                                     )

            else:
                self.File_deconv.plot(navigator='auto', 
                                      navigator_kwds=dict(colorbar=True, 
                                                          scalebar_color='black',
                                                          cmap=cmap,
                                                          axes_ticks=True
                                                         )
                                     )
        
    
    def plot_histogramm(self):
        """
        Plotting the histogram of the spectrum image.
        """
        if (self.deconv == True):
            title      = self.File.metadata.General.title
            title_corr = self.check_underscores_in_title(title)

            self.File.metadata.General.title = title_corr
            self.File.get_histogram().plot()
        else:
            title      = self.File_deconv.metadata.General.title
            title_corr = self.check_underscores_in_title(title)

            self.File_deconv.metadata.General.title = title_corr
            self.File_deconv.get_histogram().plot()
            
    
    def plot_model(self,
                   navigator       = 'auto',
                   show_components = False,
                   cmap            = 'coolwarm'
                  ):
        """
        Plotting the fitting model of the spectrum image.
        
            navigator:       If True - adding a navigator instead of the navigation
                                       image
            
            show_components: If True - additionally plot each component of the model
                                       seperately
        """
        self.Fit_Model.plot(navigator       = navigator,
                            plot_components = show_components,
                            navigator_kwds=dict(colorbar=True, 
                                                scalebar_color='black',
                                                cmap=cmap,
                                                axes_ticks=True
                                               )
                           )
        

    def plot_parameter_maps(self,
                            marker_width       = 3.,
                            rotate             = False,
                            first_plasmon_only = False,
                            cmap               = 'coolwarm',
                            overview           = True
                           ):
        """
        Visualize all parameter maps that were previously generated.
        
            first_plasmon_only: If True - only the first Plasmon peak component 
                                          is plotted
                                          
            cmap: Changing the style of the colorbar by following matplotlibs styles
            
            overview:           If True - Plot every parameter map in a single
                                          figure as an overview instead of plotting
                                          each parameter map seperately
        """
        if (self.attr_deconv == False and first_plasmon_only == False):
            
            if (overview == True):
                
                if (rotate == True):
                    images = hs.stack([hs.signals.Signal2D(np.transpose(self.param_dict[title].data)
                                                          )
                                       for title in self.param_dict]
                                     ) 
                else:
                    images = hs.stack([self.param_dict[title] 
                                       for title in self.param_dict])
                
                
                vmin, vmax = [], []
                for image in images:
                    self.delete_marker(image)
                    
                    vmin.append(np.nanmean(image.data)
                                -np.nanstd(image.data)
                               )
                    vmax.append(np.nanmean(image.data)
                                +np.nanstd(image.data)
                               )
                
                hs.plot.plot_images(images, 
                                    cmap=cmap,
                                    colorbar='multi',
                                    centre_colormap=False,
                                    suptitle='Overview',
                                    suptitle_fontsize=16,
                                    label=[r'{} {}'.format(title.encode('unicode-escape').decode().replace('\\\\','\\'), 
                                                           'in '+self.param_dict[title].metadata.Signal.quantity
                                                          ) 
                                           for title in self.param_dict
                                          ],
                                    axes_decor='off',
                                    tight_layout=True,
                                    labelwrap=25,
                                    scalebar_color='black',
                                    vmin=vmin,
                                    vmax=vmax
                                   )
                
            else:
                for title in self.param_dict:
                    self.delete_marker(self.param_dict[title])
                    
                    vmin = (np.nanmean(self.param_dict[title].data)
                            -np.nanstd(self.param_dict[title].data)
                           )
                    vmax = (np.nanmean(self.param_dict[title].data)
                            +np.nanstd(self.param_dict[title].data)
                           )
                    
                    self.param_dict[title].plot(cmap = cmap,
                                                centre_colormap=False,
                                                label=('{} {}'.format(title.encode('unicode-escape').decode().replace('\\\\','\\'), 
                                                                      'in '+self.param_dict[title].metadata.Signal.quantity
                                                                     )
                                                      ),
                                                axes_ticks=True,
                                                scalebar_color='black',
                                                vmin=vmin,
                                                vmax=vmax
                                               )
                    
                    if (self.line != None):
                        self.draw_line_marker(self.param_dict[title], marker_width)
            
            if (self.linescan_plots != {}):
                for key in self.linescan_plots:
                    self.linescan_plots[key].show()
                    
                
        else:
            
            
            if (overview == True):
                
                if (rotate == True):
                    images = hs.stack([hs.signals.Signal2D(np.transpose(self.param_dict[title].data)
                                                          )
                                       for title in self.param_dict]
                                     ) 
                else:
                    images = hs.stack([self.param_dict[title] 
                                       for title in self.param_dict])
                
                vmin, vmax = [], []
                for image in images:
                    
                    vmin.append(np.nanmean(image.data)
                                -np.nanstd(image.data)
                               )
                    vmax.append(np.nanmean(image.data)
                                +np.nanstd(image.data)
                               )
                
                hs.plot.plot_images(images, 
                                    cmap=cmap,
                                    colorbar='multi',
                                    centre_colormap=False,
                                    suptitle='Overview',
                                    suptitle_fontsize=16,
                                    label=[r'{} {}'.format(title.encode('unicode-escape').decode().replace('\\\\','\\'), 
                                                           'in '+self.param_dict[title].metadata.Signal.quantity
                                                          ) 
                                           for title in self.param_dict
                                          ],
                                    axes_decor='off',
                                    tight_layout=True,
                                    labelwrap=25, 
                                    scalebar_color='black',
                                    vmin=vmin,
                                    vmax=vmax
                                   )
                
                
            else:
                for title in self.param_dict:
                    self.delete_marker(self.param_dict[title])
                    
                    vmin = (np.nanmean(self.param_dict[title].data)
                            -np.nanstd(self.param_dict[title].data)
                           )
                    vmax = (np.nanmean(self.param_dict[title].data)
                            +np.nanstd(self.param_dict[title].data)
                           )
                    
                    self.param_dict[title].plot(cmap = cmap,
                                                centre_colormap=False,
                                                label=('{} {}'.format(title.encode('unicode-escape').decode().replace('\\\\','\\'), 
                                                                      'in '+self.param_dict[title].metadata.Signal.quantity
                                                                     )
                                                      ),
                                                axes_ticks=True,
                                                scalebar_color='black',
                                                vmin=vmin,
                                                vmax=vmax
                                               )
                    
                    if (self.line != None):
                        self.draw_line_marker(self.param_dict[title], marker_width)
                    
                    self.delete_marker(self.param_dict[title])
                    
            if (self.linescan_plots != {}):
                for key in self.linescan_plots:
                    self.linescan_plots[key].show()
    
    
    def print_stats(self):
        """
        Printing standard file information.
        """
        print('Statistics of loaded spectrum image:')
        self.File.print_summary_statistics()
        print('Statistics of deconvolved spectrum image:')
        self.File_deconv.print_summary_statistics()
    
    
    def print_param_stats(self):
        """
        Printing standard parameter information
        """
        for title in self.param_dict:
            print(title)
            self.param_dict[title].print_summary_statistics()
    
    
    
    def create_dirs(self):
        """
        Creating directories and sub-directories for evaluation storage.
        """
        for directory in self.dir_list:
            self.mk_dir(os.getcwd() + os.sep + directory)
    
    def mk_dir(self, directory):
        """
        creating a directory and intermediate missing directories
        """
        if not os.path.exists(directory):
            
            print('Creating:', os.getcwd() + os.sep + directory)
            if (directory.endswith(os.sep)):
                os.makedirs(directory)
            else:
                os.makedirs(directory + os.sep)
        
    
    def select_directory(self):
        """
        select a directory and make it the current working directory
        """
        dir_name = askdirectory() # asks user to choose a directory
        self.mk_dir(dir_name)
        
        if (dir_name.endswith(os.sep)):
            os.chdir(dir_name) # changes your current directory
        else:
            os.chdir(dir_name + os.sep) # changes your current directory ending with seperator
    
    
    def delete_marker(self, im):
        try:
            del im.metadata.Markers.line_segment
            del im.metadata.Markers.line_segment1
            del im.metadata.Markers.line_segment2
        except:
            pass
    
    def draw_line_marker(self, im, marker_width):
        """
        function marking a linescan region in image by drawing
        a main line and two additional smaller lines for the indication
        of the linewidth
        """
        scale  = im.axes_manager.signal_axes[0].scale
        offset = im.axes_manager.signal_axes[0].offset

        string=str(self.line)
        line_params=string.split('(')[1].split(')')[0].split(',',-1)
        coords=[float(line_params[i].split('=')[1]) for i in range(len(line_params))]

        offset_x=coords[4]/2*np.sin(np.pi*self.line.angle()/180)
        offset_y=-coords[4]/2*np.cos(np.pi*self.line.angle()/180)

        marker    = hs.plot.markers.line_segment(
            x1=coords[0], x2=coords[2], 
            y1=coords[1], y2=coords[3],
            linewidth=marker_width*scale, color='black', linestyle='-'
        )
        marker_l  = hs.plot.markers.line_segment(
            x1=coords[0]-offset_x, x2=coords[2]-offset_x, 
            y1=coords[1]-offset_y, y2=coords[3]-offset_y,
            linewidth=marker_width/2*scale, color='black', linestyle='--'
        )
        marker_r  = hs.plot.markers.line_segment(
            x1=coords[0]+offset_x, x2=coords[2]+offset_x, 
            y1=coords[1]+offset_y, y2=coords[3]+offset_y,
            linewidth=marker_width/2*scale, color='black', linestyle='--'
        )
        
        im.add_marker(marker, permanent=True)
        im.add_marker(marker_l, permanent=True)
        im.add_marker(marker_r, permanent=True)
        
        return
    
    
    def save_evaluation(self,
                        dpi=300, 
                        fileformat='png',
                        marker_width=3.,
                        cmap='coolwarm',
                        rotate=False
                       ):
        
        """
        Saving function to save all generated parameter maps that were previously generated.
            
            dpi: Resolution with which the parameters are saved
            
            fileformat: Choose the file format (take a look at hyperspy's documentation for
                                                supported file formats)
            
            cmap: Changing the style of the colorbar by following matplotlibs styles 
        """
        self.select_directory()
        self.create_dirs()
        
        dir_fname = {r'Plasmon peak - $E_{\max}$'          : 'FPP_Emax',
                     r'Plasmon peak - $\Gamma$'            : 'FPP_FWHM',
                     r'Plasmon peak - intensity'           : 'FPP_Int' ,
            
                     r'Zero Loss peak - $E_{\max}$'        : 'ZLP_Emax',
                     r'Zero Loss peak - $\Gamma$'          : 'ZLP_FWHM',
                     r'Zero Loss peak - intensity'         : 'ZLP_Int' ,
                     
                     r'second Plasmon peak - $E_{\max}$'   : 'SPP_Emax',
                     r'second Plasmon peak - $\Gamma$'     : 'SPP_FWHM',
                     r'second Plasmon peak - intensity'    : 'SPP_Int' ,
                     
                     r'Plasmon energy - $E_{p}(q=0)$'      : 'Ep_q0',
                     r'intensity ratio - $I_{pp}/I_{zlp}$' : 'Intensity_Ratio',
                     r'thickness by log-ratio'             : 'thickness'
                    }
        
        print('Writing parameter maps to disk...')
        
        if (rotate == True):
            images = hs.stack([hs.signals.Signal2D(np.transpose(self.param_dict[title].data)
                                                  )
                              for title in self.param_dict]
                             ) 
        else:
            images = hs.stack([self.param_dict[title] 
                               for title in self.param_dict])

        vmin, vmax = [], []
        for image in images:
            vmin.append(np.nanmean(image.data)
                        -np.nanstd(image.data)
                       )
            vmax.append(np.nanmean(image.data)
                        +np.nanstd(image.data)
                       )
            
        hs.plot.plot_images(images, 
                            no_nans = True,
                            cmap = cmap,
                            colorbar='multi',
                            centre_colormap=False,
                            suptitle = 'Parameter Overview',
                            suptitle_fontsize = 16,
                            label = [r'{} {}'.format(title, 
                                                     'in '+self.param_dict[title].metadata.Signal.quantity
                                                    )
                                     for title in self.param_dict
                                    ],
                            axes_decor = 'off',
                            tight_layout = True,
                            labelwrap = 25,
                            vmin=vmin,
                            vmax=vmax
                           )
        fname = os.getcwd() + os.sep + 'overview'
        plt.savefig('overview', dpi=dpi, extension=fileformat)
        plt.close()

        for title in self.param_dict:
            self.delete_marker(self.param_dict[title])
            
            vmin = (np.nanmean(self.param_dict[title].data)
                    -np.nanstd(self.param_dict[title].data)
                   )
            vmax = (np.nanmean(self.param_dict[title].data)
                    +np.nanstd(self.param_dict[title].data)
                   )
            
            fname = os.getcwd() + os.sep + self.dir_list[0] + os.sep + dir_fname[title]
            self.param_dict[title].metadata.General.title = title
            self.param_dict[title].plot(scalebar_color='black', 
                                        cmap=cmap, 
                                        centre_colormap=False,
                                        vmin=vmin,
                                        vmax=vmax
                                       )
            if (self.line != None):
                self.draw_line_marker(self.param_dict[title], marker_width)
                    
            plt.savefig(fname, dpi=dpi, extension=fileformat)
            plt.close()
            self.delete_marker(self.param_dict[title])
        
        if (self.linescan_plots != {}):
            print('Writing line scans to disk...')
        
            for key in self.linescan_plots:
                if ('thickness' in key):
                    fname = os.getcwd() + os.sep + self.dir_list[1] + os.sep + key
                    self.linescan_plots[key].savefig(fname, dpi=dpi, extension=fileformat)
                    plt.close(self.linescan_plots[key].number)
                else:
                    if ('intensity ratio - $I_{pp}/I_{zlp}$' in key):
                        key_adj = key.split('-',-1)[0] + key.split('-',-1)[2]
                        fname = os.getcwd() + os.sep + self.dir_list[0] + os.sep + key_adj
                        self.linescan_plots[key].savefig(fname, dpi=dpi, extension=fileformat)
                        plt.close(self.linescan_plots[key].number)
                    else:
                        fname = os.getcwd() + os.sep + self.dir_list[0] + os.sep + key
                        self.linescan_plots[key].savefig(fname, dpi=dpi, extension=fileformat)
                        plt.close(self.linescan_plots[key].number)
        
        else:
            print('No line scans were found. Please remember to use the\n'
                  +'generate_linescans() attribute previous to save_evaluation()'
                  +'attribute, if line scans are demanded. Continue...'
                 )
        
        print('Writing goodness to disk...')
        title      = self.red_Chisq.metadata.General.title
        title_corr = self.check_underscores_in_title(title)
        self.red_Chisq.metadata.General.title = title_corr
        
        fname = os.getcwd() + os.sep + 'Model_red_Chisq'
        self.red_Chisq.metadata.General.title = r'adj. $\chi_{\nu}^2$ - Goodness'
        self.red_Chisq.plot(scalebar_color='black', cmap=cmap, centre_colormap=False)
        plt.savefig(fname, dpi=dpi, extension=fileformat)
        plt.close()
        
        print('Trying to access thickness signal...')
        if (self.thickness_map != None):
            print('Writing thickness estimation to disk...')
            title      = self.thickness_map.metadata.General.title
            title_corr = self.check_underscores_in_title(title)
            self.thickness_map.metadata.General.title = title_corr
            
            fname = os.getcwd() + os.sep + self.dir_list[1] + os.sep + 'thickness_logratio'
            self.thickness_map.plot(scalebar_color='black', cmap=cmap, centre_colormap=False)
            plt.savefig(fname, dpi=dpi, extension=fileformat)
            plt.close()
        
        else:
            print('No thickness signal was found. Please calculate the thickness'
                  +'by log-ratio if demanded.')
        
        print('Finished!')
        
    
    # Calculate thickness by log ratio of ZLP-Area to Total Area of EELS-data
    def calc_thickness(self, elements, concentrations):
        """
        Estimation of the sample thickness by the Log-Ratio method
                
            LATEX Formula:
                t = \lambda \ln{ \frac{I_t}{I_0} }
                
                with:
                         I_t = total intensity
                    and  I_0 = elastically scattered intensity (0-fold intensity)
        
        absolute accuracy at +- 20% for inorganic specimen 
        
        For more information see:
            EELS Log-Ratio Technique for Specimen-Thickness
            Measurement in the TEM                    
        
            MALIS, S.C. CHENG, AND R.F. EGERTON
            JOURNAL OF ELECTRON MICROSCOPY TECHNIQUE 8:193-200 11988)
        
        Mean Free Path (\lambda) estimation is automated for a given elemental 
        composition.
        
        IMPORTANT:
            MFP-Automation takes metadata of file as input. Please verify that 
            values for beam energy and collection angle specified in .dm3/.dm4 
            files are correct before using this function.
        
        Currently supported elements: 
        (
         Ag, Al, Au, Be, Ca, Ce, Cu, Dy, Er, Eu, Fe, Gd, Ho, La, Lu, Mg, Nb, 
         Nd, Ni, P, Pd, Pm, Pr, Sm, Sn, Tb, Ti, Y, Yb, Zn, Zr
        )
        """
        if (self.is_lazy == True):
            self.File.compute()
            t_lambda = self.File.estimate_thickness(threshold=self.elastic_threshold).T
            
        mfp                = self.estimate_MFP(elements, concentrations)
        print('Estimated mean free path: %.2E nm' % (mfp,))
        self.thickness_map = t_lambda * mfp
        
        self.thickness_map.metadata.General.title = r'thickness by log-ratio' 
        self.thickness_map.metadata.Signal.quantity = 'nm'
        
        #as the absolute error is about 20 % following Malis et. al
        #we assume this to be the gain factor
        var = self.thickness_map.estimate_poissonian_noise_variance(gain_factor=0.20)
        self.thickness_map.metadata.Signal.Noise_properties.variance = var
        
        self.param_dict.update( {self.thickness_map.metadata.General.title : self.thickness_map} )
        
        if (self.is_lazy == True):
            self.File = self.File.as_lazy()
    
    
    def estimate_MFP(self, elements, concentrations):#, elements, concentration):
        """
        Estimation of the Mean Free Path for a given elemental composition (in nm).
        
        IMPORTANT:
            MFP-Automation takes metadata of file as input. Please verify that 
            values for beam energy and collection angle specified in .dm3/.dm4 
            files are correct before using this function.
            Default values (if not specified in metadata):
                E0   = 300 keV
                beta =  10 mrad
        
        Currently supported elements: 
        (
         Ag, Al, Au, Be, Ca, Ce, Cu, Dy, Er, Eu, Fe, Gd, Ho, La, Lu, Mg, Nb, 
         Nd, Ni, P, Pd, Pm, Pr, Sm, Sn, Tb, Ti, Y, Yb, Zn, Zr
        )
        """
        
        element_dict = {'Ag' : mend.Ag,
                        'Al' : mend.Au,
                        'Au' : mend.Au,
                        'Be' : mend.Be,
                        'Ca' : mend.Ca,
                        'Ce' : mend.Ce,
                        'Cu' : mend.Cu,
                        'Dy' : mend.Dy,
                        'Er' : mend.Er,
                        'Eu' : mend.Eu,
                        'Fe' : mend.Fe,
                        'Gd' : mend.Gd,
                        'Ho' : mend.Ho,
                        'La' : mend.La,
                        'Lu' : mend.Lu,
                        'Mg' : mend.Mg,
                        'Nb' : mend.Nb,
                        'Nd' : mend.Nd,
                        'Ni' : mend.Ni,
                        'P'  : mend.P,
                        'Pd' : mend.Pd,
                        'Pm' : mend.Pm,
                        'Pr' : mend.Pr,
                        'Pt' : mend.Pt,
                        'Sm' : mend.Sm,
                        'Sn' : mend.Sn,
                        'Tb' : mend.Tb,
                        'Ti' : mend.Ti,
                        'Tm' : mend.Tm,
                        'Y'  : mend.Y,
                        'Yb' : mend.Yb,
                        'Zn' : mend.Zn,
                        'Zr' : mend.Zr
                       }
        
        const              = 0.3 # Malis et. al - konstante r nach eq.(4)
        number_of_elements = len(elements)
        fi_Zi_numerator    = 0
        fi_Zi_denominator  = 0

        for i in range(number_of_elements):
            
            Z                  = element_dict[elements[i]].atomic_number
            fi_Zi_numerator   += concentrations[i] * Z**(1+const)
        
        
        for i in range(number_of_elements):
            
            Z                  = element_dict[elements[i]].atomic_number
            fi_Zi_denominator += concentrations[i] * Z**(const)

        Z_eff = fi_Zi_numerator/fi_Zi_denominator

        m    = 0.36 # Malis EELS paper - exponent m nach eq.(8)
        E_m  = 7.6*Z_eff**m # eq.(8): 7.6 eV
        
        E_0  = self.File.metadata.Acquisition_instrument.TEM.beam_energy
        beta = self.File.metadata.Acquisition_instrument.TEM.Detector.EELS.collection_angle

        F = (1 + E_0/1022)/(1 + E_0/511)**2 # Malis EELS paper - E_0 in keV nach eq.(6) 

        mean_free_path = 106 * F * E_0 / (E_m * np.log(2 * beta * E_0 / E_m))
        
        return mean_free_path
    
        
    def line_roi(self, 
                 param_map,
                 order,
                 width,
                 interactive=False,
                 cmap='coolwarm'
                ):
        """
        Comments missing
        """
        if (self.line == None):
            x1 = 0 
            y1 = 0 
            x2 = 15 
            y2 = 15
            
            self.line = hs.roi.Line2DROI(x1 * param_map.axes_manager[0].scale + 
                                         param_map.axes_manager[0].offset,
                                         y1 * param_map.axes_manager[1].scale + 
                                         param_map.axes_manager[1].offset,
                                         x2 * param_map.axes_manager[0].scale + 
                                         param_map.axes_manager[0].offset,
                                         y2 * param_map.axes_manager[1].scale + 
                                         param_map.axes_manager[1].offset,
                                         linewidth = width 
                                        )
        
        if (interactive == True):
            param_map.plot(scalebar_color='black', cmap=cmap, centre_colormap=False)
            
            time = self.time
            answer = False
            print('Waiting %d seconds for adjustment. ' % (time,))
            while (answer == False):
                plt.ion()
                line = self.line.interactive(param_map, order=order)
                plt.show()
                plt.draw()
                plt.pause(time)
                answer = self.yes_or_no('\nTo exit adjustment, please type (y).'
                                        + 'To continue adjustment for %d, please type (n)' % (time,)
                                       )
            
            return line
            
        else:
            return self.line(param_map, order=order)

            
    def line_variance(self, 
                      param_map,
                      order,
                      line_scan
                     ):
        """
        larger image test
        angle test
        scaling of elipse effect
        interpolation
        """
        if (self.line == None):
            print('No line is defined. No calculation of standard deviation is possible.')
            return
        
        angle = self.line.angle()
        width = self.line.linewidth
        
        line_std      = line_scan.deepcopy()
        line_fstd     = line_scan.deepcopy()
        scale         = line_scan.axes_manager.signal_axes[0].scale
        size          = line_scan.axes_manager.signal_axes[0].size

        properties  = str(self.line).split('=',-1)[1::]
        line_coords = []
        for ele in properties:
            line_coords.append(ele.split(',')[::2])

        line_coords[-1][0]=line_coords[-1][0].strip(')')
        np.array(line_coords, dtype=float).ravel()

        src=np.array([line_coords[0], line_coords[1]], dtype=float)
        dst=np.array([line_coords[2], line_coords[3]], dtype=float)

        #scalar product for length if needed
        def getLength(src,dst):
            length_vec = dst-src
            length=0
            for num in length_vec:
                length+=num[0]*num[0]
            return np.sqrt(length)

        #center of src and destination as well as director by vector calculation
        length = abs(getLength(src, dst))
        
        CoM = src + (dst-src) / 2
        director = (dst-src)
        director_scaled = scale * director / length

        #rotate around CoM
        def rotatePoint(point, center, angle):
            angle = angle * (np.pi/180)#convert to radians

            rotatedX = (np.cos(angle) * (point[0]-center[0]) 
                        - np.sin(angle) * (point[1]-center[1]) + center[0])
            rotatedY = (np.sin(angle) * (point[0]-center[0]) 
                        + np.cos(angle) * (point[1]-center[1]) + center[1])

            return np.array([rotatedX,rotatedY], dtype=float)

        
        #points rotated orthogonaly and scaled corresponding to width of linescan
        src_rotate = rotatePoint(src, CoM, 90)
        dst_rotate = rotatePoint(dst, CoM, 90)
        
        length_rot = abs(getLength(src_rotate, dst_rotate))
        
        director_rot        = (dst_rotate-src_rotate) 
        director_rot_scaled = scale * director_rot / length_rot
        
        src_rotate_scaled   = CoM - director_rot_scaled * width / 2
        dst_rotate_scaled   = CoM + director_rot_scaled * width / 2
        
        
        for step in range(size):
            #Stepping over line scan positions orthogonaly to estimate each variance
            #of the linewidth line scan data point (CoM and scaled director
            #shift the coordinates correspondingly
            offset = director / 2
            x1 = src_rotate_scaled[0] + director_scaled[0] * step - offset[0]
            y1 = src_rotate_scaled[1] + director_scaled[1] * step - offset[1]
            x2 = dst_rotate_scaled[0] + director_scaled[0] * step - offset[0]
            y2 = dst_rotate_scaled[1] + director_scaled[1] * step - offset[1]
            
            line_rot = hs.roi.Line2DROI(x1=x1, 
                                        y1=y1, 
                                        x2=x2, 
                                        y2=y2, 
                                        linewidth=scale
                                       )

            #filter zeros and set zero std values to mean std by spatial std analysis
            rotated_line = line_rot(param_map, order=order)

            line_nozeros = np.where(rotated_line.data != 0,
                                    rotated_line.data,
                                    np.nan
                                   )
            std = np.nanstd(line_nozeros.data)
            
            line_std.isig[step] = std

        
        mean_line = np.nanmean(line_std.data)
        
        line_var  = hs.signals.Signal1D(np.where(line_std.data != 0., 
                                                 line_std.data**2,
                                                 mean_line**2
                                                )
                                       )

        print('Variance estimation for:', param_map.metadata.General.title)
        print('mean variance: ',np.nanmean(line_var.data))
        
        try:
            #Alternatively: take param_map-variance property using fitting error estimation by:
            
            for step in range(size):
                #Stepping over line scan positions orthogonaly to estimate each variance
                #of the linewidth line scan data point (CoM and scaled director
                #shift the coordinates correspondingly)
                offset = director / 2
                x1 = src_rotate_scaled[0] + director_scaled[0] * step - offset[0]
                y1 = src_rotate_scaled[1] + director_scaled[1] * step - offset[1]
                x2 = dst_rotate_scaled[0] + director_scaled[0] * step - offset[0]
                y2 = dst_rotate_scaled[1] + director_scaled[1] * step - offset[1]

                line_rot = hs.roi.Line2DROI(x1=x1, 
                                            y1=y1, 
                                            x2=x2, 
                                            y2=y2, 
                                            linewidth=scale
                                           )

                #filter zeros and set zero std values to mean std by spatial std analysis
                rotated_line_fstd = line_rot(np.sqrt(param_map.metadata.Signal.Noise_properties.variance),
                                             order=order
                                            )

                rotated_line_nonzero_fstd = np.where(rotated_line_fstd.data == 0.,
                                                     np.nan,
                                                     rotated_line_fstd.data
                                                    )

                mean_fstd = np.nanstd(rotated_line_nonzero_fstd)

                line_fstd.isig[step] = mean_fstd


            mean_line_fstd = np.nanmean(line_fstd.data)

            line_var_fstd  = hs.signals.Signal1D(np.where(line_fstd.data != 0, 
                                                          line_fstd.data**2,
                                                          mean_line_fstd**2
                                                         )
                                                )
            print('mean variance of fitted parameters: ',np.nanmean(line_var_fstd.data))
            
        except AttributeError as err:
            print(line_scan.metadata.General.title + ': \n')
            print(err)
            print('\nFalling back using spatial error estimation.')
            line_var_fstd = hs.signals.Signal1D(np.full(np.shape(line_scan.data),
                                                        np.nan
                                                       )
                                               )
        
        return line_var, line_var_fstd
            
    
    def rect_roi(self, 
                 param_map,
                 interactive=False,
                 width=15,
                 height=15,
                 cmap='coolwarm'
                ):
        """
        rectangle roi to evaluate regions of interest by simple statistical analysis
        investigating the mean and the standard deviation of the region.
        
        can also be used to investigate the variance property of a parameter map
        to further investigate the region of interest correspondingly.
        
        returns a full image of the region of investigation, which can also be analyzed
        further by e.g. using generate_linescan() in combination with estimate_pshif()
        attributes to generate parameter shifts of larger images in more detail.
        """
        if (self.roi == None):
            left   = 0 
            top    = 0 
            right  = left+width
            bottom = top+height
            
            self.roi = hs.roi.RectangularROI(param_map.axes_manager.signal_axes[0].offset +
                                             left   * param_map.axes_manager.signal_axes[0].scale,
                                             param_map.axes_manager.signal_axes[1].offset +
                                             top    * param_map.axes_manager.signal_axes[1].scale,
                                             param_map.axes_manager.signal_axes[0].offset +
                                             right  * param_map.axes_manager.signal_axes[0].scale,
                                             param_map.axes_manager.signal_axes[1].offset +
                                             bottom * param_map.axes_manager.signal_axes[1].scale
                                            )
        
        if (interactive == True):
            param_map.plot(scalebar_color='black', cmap=cmap)
            answer = False
            while (answer == False):
                plt.ion()
                roi = self.roi.interactive(param_map)
                plt.show()
                plt.draw()
                plt.pause(30)
                answer = self.yes_or_no('If finished: please type (y)' + '\n' + 
                                        'To continue adjustment: pleas type (n).'
                                       )
                
            print('Mean: ', roi.mean(axis=(0,1)).data)
            print('Std: ', roi.std(axis=(0,1)).data)
            
            return roi
            
        else:
            print('Mean: ', roi.mean(axis=(0,1)).data)
            print('Std: ', roi.std(axis=(0,1)).data)
            
            return self.roi(param_map)
            
            
    def generate_linescan(self,
                          param_map,
                          order,
                          number,
                          width,
                          interactive = False,
                          cmap        = 'coolwarm',
                          std_by_fitting = False
                         ):
        """
        Generat
        """
        line_scan = self.line_roi(param_map, 
                                  order,
                                  interactive = interactive, 
                                  width       = width,
                                  cmap        = cmap
                                 )
        
        line_var, line_var_fstd  = self.line_variance(param_map, order, line_scan)
        if (std_by_fitting == True and np.isfinite(line_var_fstd.data).all()):
            line_scan.metadata.Signal.set_item("Noise_properties.variance", 
                                               line_var_fstd
                                              )
        else:
            line_scan.metadata.Signal.set_item("Noise_properties.variance", 
                                               line_var
                                              )
        if (number == None):
            line_scan.metadata.General.title = (param_map.metadata.General.title + 
                                                str(' - line scan')
                                               )
        else:
            line_scan.metadata.General.title = (param_map.metadata.General.title + 
                                                str(' - line scan' + str(number))
                                               )
        
        self.linescans[param_map.metadata.General.title] = line_scan
        
    
    
    # Calculate parametershifts by gausfitting
    def generate_linescans(self,
                           order              = 0,
                           parameter_shifts   = False,
                           polygradn          = 2,
                           try_force_positive = False,
                           try_force_negative = False,
                           slope_th           = 0.,
                           cmap             = 'coolwarm',
                           interactive      = True,
                           show             = False,
                           position         = 1.,
                           sensitivity      = 3,
                           maxpeakn         = 10,
                           medfilt_radius   = 3,
                           peakgroup        = 5,
                           number           = None,
                           std_by_fitting   = False
                          ):
        """
        generating line scans and if demanded try estimating the parameter shifts
        by using a simple gaussian in addition to a polynomial fit for estimation
        of background influences like thickness variations in the specimen
        
        order - Interpolation type used for generating linescan (0: nearest,
                1: linear, 2:cubic)
        
        parameter_shifts - if True automatic parameter shift estimation is applied
                           using O'Haver's gaussian peak parameter estimation for
                           parameter initialization and fitting the polynom using
                           numpy.polyfit as well as scipy.optimize.curve_fit with
                           underlying weighted least squared minimization for the
                           polynomial fit in addition to the gaus fit. 
                           Strong bounds are applied to the fitting routine to 
                           stabilize the estimation. The weights for the polynomial
                           fit are chosen to slightly enhance the contribution of 
                           the data points on the edges of a linescan. 
                           Standard weights are used afterwards for the 
                           gaussian fit on the substracted data as the initial
                           parameters will fit the data quite well. The superposition
                           of both function will then be optimized by a weighted
                           least squared minimization adjusting polynomial and
                           gaussian parameters accordingly.
        
        
        polygradn - Polynomial degree (P(x)= a+b*x+c*x**2+...)
          
        try_force_positive/negative - Try forcing only positive or negative parameter
                                      shift estimations by disabling the O'Haver
                                      estimation for the inverse peak estimation
                                      and setting bounds for peak height accordingly
                                      positive or negative.
        
        interactive - If True the first parameter map 'E_p(q=0)' will be set
                      to adjust the line scan. Default is set to True as the
                      starting class parameter self.line is not initialized
                      at the first attempt. Only recommended to set to False
                      if line scan parameters are already set to the class
                      attribute self.line (see Line2DROI in hyperspy doc for
                                           further information)
                                           
        show - If True plotting the resulting optimized function to the linescan.
               Recommended to be False as results are saved in self.linescan_plots
               dictionary and plotting all line scans to seperate figures takes
               some time for matplotlib
               
        position - Does nothing yet
        
        sensitivity - Scaling factor of minimum sigma condition of bounds for
                      least squared minimization
                      
        maxpeakn,medfilt_radius,peakgroup
          maxpeakn  - Maximum peaks that are estimated O'Haver algorythm,
          medfilter - median filter for smoothing line scan (scipy medfilter),
          peakgroup - Maximum neighbouring points taken for peak height est.
                      See hyperspy documentation for further information on 
                      O'Haver peak estimation
        
        number - if specified the number will be written to the figure title
                 of the corresponding line scan to enumerate the line scans
                 for on SI evaluation. Attention: not supported fully yet!
                 
        std_by_fitting - If True taking parameter uncertainties by fitting
                         estimation of the calculated Fit_Model to estimate
                         line scan standard deviation property instead of 
                         spatial standard deviation
        """
        res = [param_map for param_map in list(self.param_dict.keys()) 
               if 'Plasmon energy - $E_{p}(q=0)$' in param_map
              ]
        
        width = 5 * abs(self.param_dict[res[0]].axes_manager[0].scale)
        estimate_shift  = [r'thickness by log-ratio',
                           r'Plasmon peak - $E_{\max}$', 
                           r'Plasmon peak - $\Gamma$',
                           r'Plasmon peak - intensity', 
                           r'Zero Loss peak - intensity',
                           r'Plasmon energy - $E_{p}(q=0)$', 
                           r'intensity ratio - $I_{pp}/I_{zlp}$'
                          ]
        
        thickness = estimate_shift[0]
        
        if (parameter_shifts == True):
            if (self.line == None):
                self.generate_linescan(self.param_dict[res[0]],
                                       order,
                                       number,
                                       width,
                                       interactive = True,
                                       cmap = cmap,
                                       std_by_fitting = std_by_fitting
                                      )
                
            else:
                self.generate_linescan(self.param_dict[res[0]],
                                       order,
                                       number,
                                       width,
                                       interactive = False,
                                       cmap = cmap,
                                       std_by_fitting = std_by_fitting
                                      )
                
            
                
            self.estimate_parametershift(
                self.linescans[res[0]],
                show,
                force_positive = try_force_positive,
                force_negative = try_force_negative,
                slope_th       = slope_th,
                position    = position,
                sensitivity = sensitivity,
                maxpeakn    = maxpeakn,
                medfilt     = medfilt_radius,
                peakgroup   = peakgroup,
                poly_gradn  = polygradn
            )
            
            if (self.thickness_map != None):
                self.generate_linescan(self.param_dict[thickness],
                                       order,
                                       number,
                                       width,
                                       interactive = False,
                                       cmap = cmap,
                                       std_by_fitting = std_by_fitting
                                      )

                self.estimate_parametershift(
                    self.linescans[thickness],
                    show,
                    force_positive = try_force_positive,
                    force_negative = try_force_negative,
                    slope_th       = slope_th,
                    position    = position,
                    sensitivity = sensitivity,
                    maxpeakn    = maxpeakn,
                    medfilt     = medfilt_radius,
                    peakgroup   = peakgroup,
                    poly_gradn  = polygradn,
                    only_poly   = True
                )

            for param_title in self.param_dict:
                if (param_title not in res and param_title != thickness):
                    self.generate_linescan(self.param_dict[param_title],
                                           order,
                                           number,
                                           self.line.linewidth,
                                           interactive = False,
                                           std_by_fitting = std_by_fitting
                                          )

                    if (param_title in estimate_shift):
                        self.estimate_parametershift(
                            self.linescans[param_title],
                            show,
                            force_positive = try_force_positive,
                            force_negative = try_force_negative,
                            slope_th       = slope_th,
                            position    = position,
                            sensitivity = sensitivity,
                            maxpeakn    = maxpeakn,
                            medfilt     = medfilt_radius,
                            peakgroup   = peakgroup,
                            poly_gradn  = polygradn
                        )
                        
                    else:
                        offset = self.linescans[param_title].axes_manager.signal_axes[0].offset
                        size = self.linescans[param_title].axes_manager.signal_axes[0].size 
                        scale = self.linescans[param_title].axes_manager.signal_axes[0].scale
                        max_pos = offset + scale * size

                        linescan_pos         = np.linspace(offset, max_pos, size)
                        linescan_func_pos    = np.linspace(offset, max_pos, size*100)
                        
                        fig=self.plot_pshift(self.linescans[param_title],
                                             linescan_pos,
                                             linescan_func_pos,
                                             np.full(3, 0.),
                                             None,
                                             np.full(polygradn, 0.),
                                             None,
                                             show=show
                                            ) 

                        key_scan = {self.linescans[param_title].metadata.General.title : fig}
                        
                        self.linescan_plots.update(key_scan)
            
                
        if (parameter_shifts == False):
            if (self.line == None):
                self.generate_linescan(self.param_dict[res[0]],
                                       order,
                                       number,
                                       width,
                                       interactive = True,
                                       cmap = cmap
                                      )
                
            else:
                self.generate_linescan(self.param_dict[res[0]],
                                       order,
                                       number,
                                       width,
                                       interactive = False,
                                       cmap = cmap
                                      )
                
            fig=self.plot_pshift(self.linescans[res[0]],
                                 linescan_pos,
                                 linescan_func_pos,
                                 np.full(3, 0.),
                                 None,
                                 np.full(polygradn, 0.),
                                 None,
                                 show=show
                                ) 

            key_scan = {self.linescans[res[0]].metadata.General.title : fig}
            
            self.linescan_plots.update(key_fig)
            
            if (self.thickness_map != None):
                self.generate_linescan(self.param_dict[thickness],
                                       order,
                                       number,
                                       width,
                                       interactive = False,
                                       cmap = cmap,
                                       std_by_fitting = std_by_fitting
                                      )
            
            fig=self.plot_pshift(self.param_dict[thickness],
                                 linescan_pos,
                                 linescan_func_pos,
                                 np.full(3, 0.),
                                 None,
                                 np.full(polygradn, 0.),
                                 None,
                                 show=show
                                ) 

            key_scan = {self.linescans[thickness].metadata.General.title : fig}
            
            self.linescan_plots.update(key_fig)
            
            for param_title in self.param_dict:
                if (param_title not in res and param_title != thickness):
                    self.generate_linescan(self.param_dict[param_title],
                                           order,
                                           number,
                                           self.line.linewidth,
                                           interactive = False
                                          )
                    
                    offset = self.linescans[param_title].axes_manager.signal_axes[0].offset
                    size = self.linescans[param_title].axes_manager.signal_axes[0].size 
                    scale = self.linescans[param_title].axes_manager.signal_axes[0].scale
                    max_pos = offset + scale * size

                    linescan_pos         = np.linspace(offset, max_pos, size)
                    linescan_func_pos    = np.linspace(offset, max_pos, size*100)

                    fig=self.plot_pshift(self.linescans[param_title],
                                         linescan_pos,
                                         linescan_func_pos,
                                         np.full(3, 0.),
                                         None,
                                         np.full(polygradn, 0.),
                                         None,
                                         show=show
                                        ) 

                    key_scan = {self.linescans[param_title].metadata.General.title : fig}
                    self.linescan_plots.update(key_scan)

        return
    
    
    def plot_pshift(self,
                    linescan,
                    linescan_pos,
                    linescan_func_pos,
                    popt_gaus,
                    std_gaus,
                    popt_poly,
                    std_poly,
                    enable_fit=False,
                    show=True
                   ):
        """
        Plotting the estimated paramter shift in a figure object of matplotlib
        Adding a 2-sigma estimation of the parameter-shift location corresponding
        to the sheared region
        """
        quantity = {r'Plasmon peak - $E_{\max}$ '          : 'peak center', 
                    r'Plasmon peak - $\Gamma$ '            : 'FWHM', 
                    r'Plasmon peak - intensity '           : 'intensity', 

                    r'Zero Loss peak - $E_{\max}$ '        : 'peak center', 
                    r'Zero Loss peak - $\Gamma$ '          : 'FWHM',
                    r'Zero Loss peak - intensity '         : 'intensity',

                    r'second Plasmon peak - $E_{\max}$ '   : 'peak center', 
                    r'second Plasmon peak - $\Gamma$ '     : 'FWHM', 
                    r'second Plasmon peak - intensity '    : 'intensity',

                    r'Plasmon energy - $E_{p}(q=0)$ '      : 'plasmon energy', 
                    r'intensity ratio - $I_{pp}/I_{zlp}$ ' : 'intensity ratio',
                  
                    r'thickness by log-ratio '             : 'thickness'
                 }

        #estimate parameter shifts due to shear band influence
        def func_gaus(x, x0, height, sigma):
            return height * np.exp(- 1/(2 * sigma**2) * (x - x0)**2)

        def std_decimals(std_all, num):
            for i in range(len(std_all)):
                std = std_all[i]
                rounding = 10**(-num)
                
                if (round(std, num) == 0.0):
                    std_all[i] = std + rounding

            return np.round(std_all, num)
        
        title      = linescan.metadata.General.title.encode('unicode-escape').decode().replace('\\\\', '\\')
        linescan.axes_manager.signal_axes[0].name = 'Position'
        #title_corr = self.check_underscores_in_title(title)
        #linescan.metadata.General.title = title_corr

        
        if (not linescan.metadata.General.title.split('-',-1)[2].endswith('n')):
            number = linescan.metadata.General.title.split('-',-1)[2][-1:]

            fig = plt.figure(num = title + ' ' + number)
            plt.title(title + ' ' + number)
        else:

            fig = plt.figure(num = title)
            plt.title(title)
        
        name_x = linescan.axes_manager.signal_axes[0].name
        unit_x = linescan.axes_manager.signal_axes[0].units
        
        component = linescan.metadata.General.title.split('-',-1)[0]
        parameter = linescan.metadata.General.title.split('-',-1)[1]
        
        component.encode('unicode-escape').decode().replace('\\\\', '\\')
        parameter.encode('unicode-escape').decode().replace('\\\\', '\\')
        #if (component in list(quantity.keys())[-1]):
        #    name_y = quantity[component]
            
        #else:
        dict_entry = component + '-' + parameter
        
        name_y = quantity[dict_entry]
        
        unit_y = linescan.metadata.Signal.quantity

        plt.xlabel(name_x + ' in ' + unit_x)
        plt.ylabel(name_y + ' in ' + unit_y)

        #plt.plot(linescan_pos, 
        #         linescan.data,
        #         'rx',
        #         label=parameter
        #        )
        sigma = self.sigma_calc(linescan)
        plt.errorbar(linescan_pos, 
                     linescan.data,
                     yerr=sigma,
                     fmt='rx',
                     label=parameter
                    )
        
        plt.ylim(np.nanmin(linescan.data)-2*np.nanmax(sigma),
                 np.nanmax(linescan.data)+2*np.nanmax(sigma)
                )

        if ((np.isfinite(np.array(std_gaus,dtype=float)).all() 
            or np.isfinite(np.array(std_poly,dtype=float)).all()) 
            and enable_fit==True):
            
            if (std_gaus != None and std_poly != None):
                popt_gaus_r = np.round(popt_gaus,5)
                popt_poly_r = np.round(popt_poly,5)
                popt_all_r  = np.append(popt_gaus, popt_poly)

                std_gaus_r  = std_decimals(std_gaus, 3)
                std_poly_r  = std_poly

                mean        = np.mean(linescan.data)

                label_poly  = ''
                for i in range(len(popt_poly)):
                    values_str   = '$P_{%s}=%.3E \pm %.3E$ in ' % tuple((str(i), 
                                                                         popt_poly_r[i], 
                                                                         std_poly_r[i]
                                                                        )
                                                                       )

                    if (i == 0):
                        units_str    = ('$\mathrm{%s}$') % tuple((unit_y,))
                    else:
                        units_str    = ('$\mathrm{%s}/\mathrm{%s}^%s$') % tuple((unit_y,unit_x,str(i)))

                    p_string     = values_str+units_str
                    label_poly  += '\n'+p_string

                label_str1          = ('polynomial fit: '
                                       +label_poly
                                      )
                label_str2          = ('gaussian fit (+ mean of linescan):\n'
                                       +r'$x_0=%5.3f \pm %5.3f$ in '+unit_x+',\n'
                                       +r'$A_\mathrm{gauß}=%5.3f \pm %5.3f$ in '+unit_y+',\n'
                                       +r'$\sigma=%5.3f \pm %5.3f$ in '+unit_x
                                      )
                
                func_poly = poly.Polynomial(popt_poly)

                val_gaus  = func_gaus(linescan_func_pos, *popt_gaus)
                val_poly  = func_poly(linescan_func_pos)

                std_2 = popt_gaus[-1] * 2
                x0    = popt_gaus[0]

                ax_min   = linescan.axes_manager.signal_axes[0].offset
                ax_scale = linescan.axes_manager.signal_axes[0].scale
                ax_size  = linescan.axes_manager.signal_axes[0].size

                ax_max = ax_min + ax_scale * ax_size
                
                plt.plot(linescan_func_pos, 
                         val_poly, 
                         'g--', 
                         label=(label_str1)
                        )
                
                if ((abs(popt_gaus) > np.full(len(popt_gaus),0.0)).any()):

                    if (x0-std_2 < ax_min or x0+std_2 > ax_max):
                        label_all = ('Sum of polynomial and gaussian function\n'
                                     +' [Peak out of bounds]'
                                    )
                        

                    else:
                        plt.axvspan(x0-std_2, x0+std_2, color='red', alpha=0.2)
                        label_all = 'Sum of polynomial and gaussian function'
                    
                    tup_gaus       = [None]*(len(popt_gaus)+len(std_gaus))
                    tup_gaus[::2]  = popt_gaus_r
                    tup_gaus[1::2] = std_gaus_r

                    plt.plot(linescan_func_pos, 
                             val_gaus + mean, 
                             'b--', 
                             label=(label_str2 % tuple(tup_gaus))
                            )

                    plt.plot(linescan_func_pos, 
                             val_gaus + val_poly, 
                             'r-', 
                             label='Sum of linear and gaussian function'
                            )

        plt.legend(prop={'size': 9})


        if (show == True):
            fig.show()

        else:
            plt.close()

        return fig
        
        
    def sigma_calc(self, linescan):
        sigma         = np.sqrt(linescan.metadata.Signal.Noise_properties.variance.data)
        sigma_fin     = np.where(np.isfinite(sigma))[0]
        sigma_nonzero = np.where(np.nonzero(sigma))[0]
        
        sigma_def     = np.where(sigma_fin != sigma_nonzero)[0]
        
        sigma_nan     = np.where(np.isnan(sigma))[0]
        sigma_inf     = np.where(np.isinf(sigma))[0]
        
        sigma_undef   = np.concatenate((sigma_nonzero, sigma_nan, sigma_inf), axis=0)
        
        sigma_calc  = np.empty(len(sigma))
        
        for idx in range(len(sigma)):
            if (idx in sigma_def):
                sigma_calc[idx] = sigma[idx]
            else:
                sigma_calc[idx] = np.mean(sigma[sigma_nonzero])
        
        return sigma_calc
        
        
    def polynomial_fitting(self,
                           linescan,
                           poly_gradn
                          ):
        size                 = linescan.axes_manager.signal_axes[0].size
        scale                = linescan.axes_manager.signal_axes[0].scale
        offset               = linescan.axes_manager.signal_axes[0].offset
        max_pos              = offset + size * scale

        linescan_pos         = np.linspace(offset, max_pos, size)
        linescan_func_pos    = np.linspace(offset, max_pos, size*100)
        
        #estimate weights using gaussian distribution to weight points near 
        #linescan axis limits more strongly
        def func_gaus_distribution(x, x0, height, sigma):
            return height/(sigma*np.sqrt(2*np.pi)) * np.exp(- 1/(2 * sigma**2) * (x - x0)**2)
        
        border_bounds = np.full(size, 1.)
        for i in range(len(border_bounds)):
            border_bounds[i] = ((1-func_gaus_distribution(scale*i+offset, 
                                                          (max_pos - offset)/2, 
                                                          1., 
                                                          (max_pos - offset)*2
                                                         )
                                )
                               )
            
        print('weight factors: ', border_bounds)
        
        sigma_finite = self.sigma_calc(linescan)
        
        weights = np.full(size, np.sqrt(size)) / border_bounds
        popt_poly, pcov_poly = poly.polyfit(linescan_pos, linescan.data, poly_gradn, full=True, w=weights)
        print('\nfitted initial polynomial parameter: ', popt_poly)

        
        def func_poly(x, *popt_poly):
            poly_func = poly.Polynomial(popt_poly)
            return poly_func(x)
            
        popt_poly, pcov_poly = curve_fit(func_poly, 
                                         linescan_pos, 
                                         linescan.data,
                                         p0 = popt_poly,
                                         maxfev=10000,
                                         sigma =sigma_finite
                                        )
        
        print('\npolynomial weighted least squared fit: ', popt_poly, pcov_poly)
        return popt_poly, pcov_poly
    
    
    def arrange_peaks(self, 
                      peaks_ar, 
                      is_negative=False
                     ):
        peaks = peaks_ar[0]

        pos    = 'position'
        height = 'height'
        width  = 'width'
        nrows  = len(peaks)

        if nrows != 0:
            x0    = peaks[pos].astype(float)
            A     = peaks[height].astype(float)
            sigma = peaks[width].astype(float)

        else:
            x0     = [0.]
            A      = [0.]
            sigma  = [0.]
            nrows += 1


        if is_negative:
            peaks = [(x0[i],-1*A[i], sigma[i]) for i in range(nrows)]
        
        else:
            peaks = [(x0[i],   A[i], sigma[i]) for i in range(nrows)]

        return peaks
        
        
    def estimate_parametershift(self,
                                linescan,
                                show,
                                force_positive = False,
                                force_negative = False,
                                enable_fit  = False,
                                slope_th    = 0.,
                                position    = 1.,
                                maxpeakn    = 10,
                                medfilt     = 3,
                                peakgroup   = 10,
                                sensitivity = 3,
                                poly_gradn  = 2,
                                only_poly   = False
                               ):
        """
        Calculate a parametershift by line scan analysis using a gaussian estimation
        for signifcant shifts and a linear estimation for thickness dependence.
        """
        print('\nEstimation for: ' + linescan.metadata.General.title + '\n')
        
        #estimate parameter shifts due to shear band influence
        def func_gaus(x, x0, height, sigma):
            return height * np.exp(- 1/(2 * sigma**2) * (x - x0)**2)

        size                 = linescan.axes_manager.signal_axes[0].size
        scale                = linescan.axes_manager.signal_axes[0].scale
        offset               = linescan.axes_manager.signal_axes[0].offset
        max_pos              = offset + size * scale

        if (position == 1.):
            position == (max_pos - offset) / 2
            if (self.x0):
                position = self.x0

        else: 
            if (self.x0):
                position = self.x0
            else:
                position = position

        linescan_pos         = np.linspace(offset, max_pos, size)
        linescan_func_pos    = np.linspace(offset, max_pos, size*100)
        
        sigma_finite = self.sigma_calc(linescan)
        
        mean = np.nanmean(linescan.data)
        std  = np.nanstd(linescan.data)

        popt_poly, pcov_poly = self.polynomial_fitting(linescan, poly_gradn)
        
        poly_val             = poly.polyval(linescan_pos, popt_poly)
        
        if (only_poly == False):
            substracted          = hs.signals.Signal1D(linescan.data - poly_val)
            substracted_inv      = hs.signals.Signal1D(poly_val - linescan.data)


            # set the signal axis for both signals
            substracted.axes_manager.signal_axes     = linescan.axes_manager.signal_axes
            substracted_inv.axes_manager.signal_axes = linescan.axes_manager.signal_axes

            if ('intensity' in linescan.metadata.General.title):
                amp_thresh = std*0.01*(max_pos-offset)/2

            elif ('Gamma' in linescan.metadata.General.title):
                amp_thresh = std*0.01*(max_pos-offset)/2

            else:
                amp_thresh = std*0.01*(max_pos-offset)/2


            if (force_positive == True):
                try:
                    peaks_positive       = substracted.find_peaks1D_ohaver(maxpeakn = maxpeakn,
                                                                           medfilt_radius=medfilt,
                                                                           slope_thresh=0,
                                                                           amp_thresh=amp_thresh,
                                                                           peakgroup=peakgroup
                                                                          )
                    print('OHaver estimates: ', peaks_positive)

                except:
                    peaks_positive       = [np.array([(0.,0.,1.)],
                                                     dtype=[('position', '<f8'), ('height', '<f8'), ('width', '<f8')])]
                    
                peaks_result = self.arrange_peaks(peaks_positive, is_negative=False)

            elif (force_negative == True):
                try:
                    peaks_negative       = substracted_inv.find_peaks1D_ohaver(maxpeakn = maxpeakn,
                                                                               medfilt_radius=medfilt,
                                                                               slope_thresh=slope_th,
                                                                               amp_thresh=amp_thresh,
                                                                               peakgroup=peakgroup
                                                                              )

                    print('OHaver estimates: ', peaks_negative)

                except:
                    peaks_negative       = [np.array([(0.,0.,1.)],
                                                     dtype=[('position', '<f8'), ('height', '<f8'), ('width', '<f8')])]
                
                peaks_result = self.arrange_peaks(peaks_negative, is_negative=True)

            else:
                try:
                    peaks_positive       = substracted.find_peaks1D_ohaver(maxpeakn = maxpeakn,
                                                                           medfilt_radius=medfilt,
                                                                           slope_thresh=slope_th,
                                                                           amp_thresh=amp_thresh,
                                                                           peakgroup=peakgroup
                                                                          )
                    print('OHaver estimates: ', peaks_positive)

                except:
                    peaks_positive       = [np.array([(0.,0.,1.)],
                                                     dtype=[('position', '<f8'), ('height', '<f8'), ('width', '<f8')])]
                
                try:
                    peaks_negative       = substracted_inv.find_peaks1D_ohaver(maxpeakn = maxpeakn,
                                                                               medfilt_radius=medfilt,
                                                                               slope_thresh=slope_th,
                                                                               amp_thresh=amp_thresh,
                                                                               peakgroup=peakgroup
                                                                              )

                    print('OHaver estimates: ', peaks_negative)

                except:
                    peaks_negative       = [np.array([(0.,0.,1.)],
                                                     dtype=[('position', '<f8'), ('height', '<f8'), ('width', '<f8')])]
            

                peaks_pos = self.arrange_peaks(peaks_positive,is_negative=False)
                peaks_neg = self.arrange_peaks(peaks_negative,is_negative=True)
                peaks_result = np.concatenate((peaks_pos, peaks_neg), axis=0)
            
            result = self.check_peaks(peaks_result, position, max_pos, scale, offset, sensitivity)

            if ((result == np.array([0.,0.,1.])).all()):
                print('The parameter shifts found do not match expected ranges for gaussian estimation.'
                      +'\nMaybe sensitivity (scaling of lower bound of sigma), medfilt_radius (smoothing) or '
                      +'\npolygradn (higher DOF for thickness dependence) can lead improved peak estimation '
                      +'\nusing Ohaver.\n')

                popt_all, pcov_all = self.set_pgaus_zero(popt_poly, pcov_poly)

            else:
                # filter the most dominant peak by height to exclude broad peaks
                # taken into account by area estimation
                if (abs(result[1]) > abs(result[1])):
                    params = result
                    #params = np.round([peak[0], peak[1], peak[2]],5)
                    # trafo fwhm|sigma:  abs(1/(2*np.sqrt(2*np.log(2)))*peak[2]

                else:
                    params = result
                    #params = np.round([peak[0], -1*abs(peak[1]), peak[2]],5)
                    # trafo fwhm|sigma:  abs(1/(2*np.sqrt(2*np.log(2)))*peak[2]

                print('\nUsed estimate for most dominant gaussian peak in bounds: ', result)

                x0_min = params[0] - 2*params[2]
                x0_max = params[0] + 2*params[2]

                sigma_min = 1/2 * params[2]
                sigma_max =   2 * params[2] 

                gaus_min = params[1] - (1/2*abs(std))
                gaus_max = params[1] + (1/2*abs(std)) 

                bounds_gaus = ([x0_min, gaus_min, sigma_min],
                               [x0_max, gaus_max, sigma_max]
                              )

                #check if peak_params in bounds and correct if needed
                for i in range(len(params)):
                    if (params[i] < bounds_gaus[0][i] or params[i] > bounds_gaus[1][i]):
                        if (params[i] < bounds_gaus[0][i]):
                            params[i] = bounds_gaus[0][i]
                        elif (params[i] > bounds_gaus[1][i]):
                            params[i] = bounds_gaus[1][i]


                try:
                    popt_gaus, pcov_gaus = curve_fit(func_gaus,
                                                     linescan_pos, 
                                                     linescan.data - poly_val, 
                                                     p0 = params,
                                                     bounds = bounds_gaus,
                                                     max_nfev=10000,
                                                     sigma = sigma_finite
                                                    )

                    #length   = len(popt_poly)
                    #popt_all = np.concatenate((popt_gaus,popt_poly),axis=0)
                    #pcov_all = np.full((len(popt_all), len(popt_all)), 
                    #                   0.
                    #                  )

                    #for i in range(len(popt_all)):
                    #    for j in range(len(popt_all)):
                    #        print(i, j)
                    #        if (i < len(popt_gaus) and j < len(popt_gaus)):
                    #            pcov_all[i][j] = pcov_gaus[i][j]
                    #            print('if')
                    #        else:
                    #            pcov_all[i][j] = np.nan


                    def func_all(x, x0, height, sigma, *popt_poly):
                        poly_func = poly.Polynomial(popt_poly)

                        return height * np.exp(- 1/(2 * sigma**2) * (x - x0)**2) + poly_func(x)

                    popt_all_init        = [*popt_gaus, *popt_poly]
                    popt_all, pcov_all   = curve_fit(func_all, 
                                                     linescan_pos, 
                                                     linescan.data,
                                                     p0 = popt_all_init,
                                                     maxfev=10000,
                                                     sigma=sigma_finite
                                                    )

                except RuntimeError as err: 
                    print('Optimal parameters not found:' 
                          +str(err)
                         )

                    popt_all, pcov_all = self.set_pgaus_zero(popt_poly, pcov_poly)

                #sigma shall be positive all times
                popt_all[2] = abs(popt_all[2])

                if (np.isfinite(popt_all[2])):
                    if (self.x0_err == None):
                        #set to infinity
                        self.x0_err = np.Infinity
                    if (abs(self.x0_err) > abs(pcov_all[2][2])):
                        self.x0_err = pcov_all[2][2]
                        self.x0     = popt_all[2]
                        print('\nPosition estimation of sheared region with lower standard deviation:\n'
                              +'$%5.3f \pm %5.3f' % tuple([self.x0, self.x0_err]))

        else:
            popt_all, pcov_all = self.set_pgaus_zero(popt_poly, pcov_poly)
        
        lengaus = 3
        popt_gaus_f = popt_all[:lengaus]
        popt_poly_f = popt_all[lengaus:]

        std_gaus_f  = [pcov_all[i][i] for i in range(lengaus)]
        std_poly_f  = [pcov_all[i+lengaus][i+lengaus] for i in range(len(popt_all)-lengaus)]

        print('\nFit summarization - resulting parameters: ', popt_all)
        print('\nFit summarization - resulting covariance matrix: ', pcov_all)
        
        fig=self.plot_pshift(linescan,
                             linescan_pos,
                             linescan_func_pos,
                             popt_gaus_f,
                             std_gaus_f,
                             popt_poly_f,
                             std_poly_f,
                             enable_fit=True,
                             show=show
                            )
        
        key_scan = {linescan.metadata.General.title : fig}
        
        self.linescan_plots.update(key_scan)
        
        return

    
    def set_pgaus_zero(self, popt_poly, pcov_poly):
        popt_gaus = [0.,0.,0.]
        pcov_gaus = np.full((len(popt_gaus), 
                             len(popt_gaus)
                            ), np.inf
                           )

        length   = len(popt_poly)
        popt_all = np.concatenate((popt_gaus,popt_poly),axis=0)
        pcov_all = np.full((length+len(popt_gaus), 
                            length+len(popt_gaus)
                           ), np.inf
                          )

        for i in range(len(popt_all)):
            for j in range(len(popt_all)):
                if (i < len(popt_gaus) and j < len(popt_gaus)):
                    pcov_all[i][j] = pcov_gaus[i][j]
                else:
                    pcov_all[i][j] = np.nan

        return popt_all, pcov_all
                        
        
    def check_peaks(self, peaks, position, max_pos, scale, offset, sensitivity):
        
        #estimate parameter shifts due to shear band influence
        def func_gaus(x, x0, height, sigma):
            return height * np.exp(- 1/(2 * sigma**2) * (x - x0)**2)
        
        results  = []
        peak_old = np.array([0.,0.,1.])
        minimum  = np.array([])
        for peak in list(peaks):
            
            minimum = np.append(minimum, np.array([peak[1]]))
            if (not np.isnan(peak).any()):

                if (peak[0]+abs(peak[2]) < max_pos and peak[0]-abs(peak[2]) > offset):
                    
                    if (abs(peak[1]) >= np.nanmin(abs(minimum))):
                        
                        if (position < peak[0] + 4*abs(peak[2]) and position > peak[0] - 4*abs(peak[2])):
                            results.append(peak)
                            print('peak accepted:', peak)
            
            else:

                sigma = abs(scale * sensitivity)
                if (peak[0]+abs(sigma) < max_pos and peak[0]-abs(sigma) > offset):
                    
                    if (abs(peak[1]) >= np.nanmin(abs(minimum))):

                        if (position < peak[0] + 4*abs(sigma) and position > peak[0] - 4*abs(sigma)):
                            results.append((peak[0], peak[1], sigma))
                            print('peak accepted after ignoring bad sigma:', peak)

        if (results != []):
            
            integral = np.empty(len(results))
            for i in range(len(results)):
                integral[i] = (integrate.quad(lambda x: func_gaus(x,*results[i]), offset, max_pos)[0])

            result = results[np.where(integral==np.max(integral))[0][0]]

            peak_result = np.array([result[0],result[1],result[2]], dtype=float)
            
        else:
            peak_result = np.array([0.,0.,1.])
            
        return peak_result
    
    def cross_correlation(self,
                          param_map
                         ):
        """
        Cross correlation of HAADF image and a given Plasmon parameter image
        while both are nomalized to map values to {0, 1}.
        """
        param_map_norm  = (( self.Ep_q0.data - np.min(self.Ep_q0.data) ) / 
                           ( np.max(self.Ep_q0.data) - np.min(self.Ep_q0.data) )
                          )
        
        haadf_norm = (( self.haadf.data - np.min(self.haadf.data) ) / 
                      ( np.max(self.haadf.data) - np.min(self.haadf.data) )
                     )

        corr = correlate2d(param_map_norm, haadf_norm)

        max_corr = np.max(corr)
        mean_corr = np.mean(corr)
        indices = np.where(corr >= max_corr)

        plt.title('Correlation image of HAADF-image and Plasmon peak energy image\n' +
                  'Maximum correlation indices: ' + str(indices[0]) + str(indices[1]) + '\n' +
                  '[HAADF | Ep] - image dimensions: \n(' + str(np.shape(haadf_norm)[0]) + 
                  ', ' + str(np.shape(haadf_norm)[1]) + ')' + ',(' + str(np.shape(param_map_norm)[0]) + 
                  ', ' + str(np.shape(param_map_norm)[1]) + ')'
                 )
        
        plt.imshow(corr,
                   cmap='hot'
                  )
        
        return corr
    
    
    def plot_3D(self, 
                param_map,
                colorbar_title,
                sigma = 5
               ):
        """
        Using mayavi to generate a 3 dimensional surface plot
        Additionaly, a gaussian filter is applied to smoothen the data
            sigma: standard deviation of gaussian filter setting the level
                   of smoothening
        """
        gaus_filt = scipy.ndimage.filters.gaussian_filter(param_map.data, sigma= (sigma, sigma))
        
        mlab.surf(gaus_filt, warp_scale='auto')
        mlab.colorbar(title=colorbar_title, orientation='vertical', label_fmt='%.3f')


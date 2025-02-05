import field_distortion_read_in_files as fld
import FGCentroid as fg
import numpy as np
from astropy.io import fits

whichFile = 1
fldFile = r'C:\Users\antho\Videos\NG\PSF_Characterization\All'
fldBright = 'bright-pos_ZeropThree'
fldDark = 'dark-01022025'
fgFile = r"C:\Users\antho\Videos\NG\PSF_Characterization\All\avg_lights_sub_-pos_ZeroZero.fits"
fgArray = fits.getdata(fgFile)
M = fgArray[1005:1013,1321:1329]

if whichFile == 0:
    returns = fld.files_in(fldFile,fldBright,fldDark,4)
elif whichFile == 1:
    returns = 'Testing'

print(returns)

fg.FGCentroid2(M,pkRow=1009,pkCol=1324,Ncentr=1013-1005,Method='Gaussian',SNRthresh=1)
'''Takes into account corrected pixel positions for x and y (not z).
    in main.py:
    array = fits.getdata(filepath for avg file)
    M = array[row_start:row_end, col_start: col_end]
    To select region b/w 2-sigma and 3-sigma:  

    
    pix_corr_x -> Make np.zeros((1944,2592))
    pix_corr_y -> Make np.zeros((1944,2592))
    pkRow -> row value for brightest pixel
    pkCol -> col value for brightest pixel
    Ncentr -> row_end - row_start
    Method -> "RotGaussian"
    SNRthresh, bright_x, bright_y -> 1 for all of them
    
    '''
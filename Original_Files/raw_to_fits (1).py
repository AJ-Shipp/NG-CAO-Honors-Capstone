from astropy.io import fits
import numpy as np


if __name__ == '__main__':


    filename = 'goni_+16mdeg_bright_'
    for i in range(5):

        raw_file = filename + str(i+1) +'.raw'
        raw_imarray = np.fromfile(raw_file, dtype='int16')
        reshaped_raw_imarray = np.reshape(raw_imarray, (1, 2048, 2048))
        fits_file = filename + str(i+1) +'.fits'
        hdu = fits.PrimaryHDU(reshaped_raw_imarray)
        hdu.writeto(fits_file)
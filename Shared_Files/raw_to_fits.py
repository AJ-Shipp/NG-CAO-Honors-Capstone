from astropy.io import fits
import numpy as np
import os

if __name__ == '__main__':

    filename = r"C:\Users\antho\Videos\NG\Testing_2-7\test1_1d7ND.raw"
    # "C:\Users\antho\Videos\NG\PSF_Characterization\p_0-0\pos_ZeroZero-0.Raw"

    for j in range(0,30):
        raw_file1 = filename #+ str(j) + '.Raw'

        if os.path.isfile(raw_file1):
            raw_imarray = np.fromfile(raw_file1, dtype='int16')
            reshaped_raw_imarray = np.reshape(raw_imarray, (1944,2592))
            fits_file = raw_file1.split('.')[0]+'.fits'
            hdu = fits.ImageHDU(reshaped_raw_imarray)
            prim = fits.PrimaryHDU()
            hdul = fits.HDUList([prim,hdu])
            hdu.writeto(fits_file, overwrite=True)

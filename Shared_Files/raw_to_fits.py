from astropy.io import fits
import numpy as np
import os

if __name__ == '__main__':
    j = 0
    i = 0

    filename = 'C:/Users/antho/Videos/NG/Testing_1-2/baseDark/dark-010220251059'

    for j in range(60):
        raw_file1 = filename + str(j) + '-'

        for i in range(60):

            raw_file2 = raw_file1 + str(i) +'.Raw'
            
            if os.path.isfile(raw_file2):
                raw_imarray = np.fromfile(raw_file2, dtype='int16')
                reshaped_raw_imarray = np.reshape(raw_imarray, (1, 2592, 1944))
                fits_file = filename + str(j) + '-' + str(i) +'.fits'
                hdu = fits.PrimaryHDU(reshaped_raw_imarray)
                hdu.writeto(fits_file)
from astropy.io import fits
import numpy as np
import os

transform = 2

if __name__ == '__main__':

    filename = r"C:\Users\antho\Videos\NG\Testing_2-10\PSFdata_ff\ff_all\dark_0-0_-0210202510"

    if transform == 0:
        for i in range(0,999999):
            raw_file1 = filename + str(i)
            for j in range(0,99): 
                raw_file1 = raw_file1 + '-' + str(j) + '.raw'

                if os.path.isfile(raw_file1):
                    raw_imarray = np.fromfile(raw_file1, dtype='int16')
                    reshaped_raw_imarray = np.reshape(raw_imarray, (1944,2592))
                    fits_file = raw_file1.split('.')[0]+'.fits'
                    hdu = fits.ImageHDU(reshaped_raw_imarray)
                    prim = fits.PrimaryHDU()
                    hdul = fits.HDUList([prim,hdu])
                    hdu.writeto(fits_file, overwrite=True)

    if transform == 1:
        for j in range(0,40):
            raw_file1 = filename #+ str(j) + '.Raw'

            if os.path.isfile(raw_file1):
                raw_imarray = np.fromfile(raw_file1, dtype='int16')
                reshaped_raw_imarray = np.reshape(raw_imarray, (1944,2592,2))
                fits_file = raw_file1.split('.')[0]+'.fits'
                hdu = fits.ImageHDU(reshaped_raw_imarray[:,:,0])
                prim = fits.PrimaryHDU()
                hdul = fits.HDUList([prim,hdu])
                hdu.writeto(fits_file, overwrite=True)

    if transform == 2:
        for i in range(0,9999):
            raw_file1 = filename + str(i) + '-'
            for j in range(0,40):
                raw_file2 = raw_file1 + str(j) + '.Raw'

                if os.path.isfile(raw_file2):
                    raw_imarray = np.fromfile(raw_file2, dtype='int16')
                    reshaped_raw_imarray = np.reshape(raw_imarray, (1944,2592,2))
                    fits_file = raw_file2.split('.')[0]+'.fits'
                    hdu = fits.ImageHDU(reshaped_raw_imarray[:,:,0])
                    prim = fits.PrimaryHDU()
                    hdul = fits.HDUList([prim,hdu])
                    hdu.writeto(fits_file, overwrite=True)

print(fits_file)
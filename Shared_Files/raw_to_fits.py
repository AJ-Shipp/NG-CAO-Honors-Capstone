from astropy.io import fits
import numpy as np
import os

# def runFile(fileIn):
#     raw_imarray = np.fromfile(fileIn, dtype='int16')
#     reshaped_raw_imarray = np.reshape(raw_imarray, (1, 2592, 1944))
#     fits_file = filename + str(j) + '-' + str(i) +'.fits'
#     hdu = fits.PrimaryHDU(reshaped_raw_imarray)
#     hdu.writeto(fits_file)


if __name__ == '__main__':
    j = 0
    i = 0
    # currentStep = 0

    filename = 'C:/Users/antho/Videos/NG/PSF_Characterization/p_0-0/pos_ZeroZero-01022025150'

    for j in range(0,10000):
        raw_file1 = filename + str(j) + '-'

        for i in range(0,30):
            raw_file2 = raw_file1 + str(i) +'.Raw'

            if os.path.isfile(raw_file2):
                raw_imarray = np.fromfile(raw_file2, dtype='int16')
                reshaped_raw_imarray = np.reshape(raw_imarray, (2,1944,2592))
                fits_file = filename + str(j) + '-' + str(i) +'.fits'
                hdu = fits.PrimaryHDU(reshaped_raw_imarray)
                hdu.writeto(fits_file)


    # for i in range(0,9):
    #     if currentStep == 0:
    #         for j in range(0,999999):
    #             for k in range(0,9):
    #                 raw_file = filename + '0-0/pos_ZeroZero-01022025' + str(j) + '-' + str(k)
    #                 if os.path.isfile(raw_file):
    #                     runFile(raw_file)

    #                 k+=1
                
    #             j+=1 
        
    #     i+=1
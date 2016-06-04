import numpy as np
import math
#from scipy.signal import get_window
import dft_model as dft
#import matplotlib.pyplot as plt
#import utility_functions as UF

def stft_analysis(x, w, N, H) :
    """                                                                     
    Analysis of a sound using the short-time Fourier transform              
    x: input array sound
    w: analysis window, 
    N: FFT size, 
    H: hop size      
    
    returns xmX, xpX: magnitude and phase spectra                           
    """
    if (H <= 0):              # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")
    
    M = w.size                         # size of analysis window
    hM1 = int(math.floor((M + 1) / 2)) # half analysis window size by rounding
    hM2 = int(math.floor(M / 2))       # half analysis window size by floor
    # add zeros at beginning to center first window at sample 0
    x = np.append(np.zeros(hM2), x) 
    # add zeros at the end to analyze last sample
    x = np.append(x, np.zeros(hM2)) 
    pin = hM1           # initialize sound pointer in middle of analysis window
    pend = x.size - hM1 # last sample to start a frame
    w = w / sum(w)      # normalize analysis window
    while pin <= pend:  # while sound pointer is smaller than last sample
        # Analysis------------------------
        x1 = x[pin - hM1:pin + hM2]    # select one frame of input sound
        mX, pX = dft.dft_analysis(x1, w, N) # compute dft
        
        # Synthesis-----------------------
        if pin == hM1:          # if first frame create output arrays
            xmX = np.array([mX])
            xpX = np.array([pX])
        else:                   # append output to existing array
            xmX = np.vstack((xmX, np.array([mX])))
            xpX = np.vstack((xpX, np.array([pX])))
        pin += H            # advance sound pointer
        
    return xmX, xpX


def stft_synthesis(mY, pY, M, H) :
    """                                                                     
    Synthesis of a sound using the short-time Fourier transform             
    mY: magnitude spectra, pY: phase spectra, M: window size, H: hop-size   
    returns y: output sound                                                 
    """
    hM1 = int(math.floor((M + 1) / 2)) # half analysis window size by rounding
    hM2 = int(math.floor(M / 2))       # half analysis window size by floor
    nFrames = mY[:, 0].size            # number of frames
    y = np.zeros(nFrames * H + hM1 + hM2) # initialize output array
    pin = hM1
    
    for i in range(nFrames):                      # iterate over all frames
        y1 = dft.dftSynth(mY[i, :], pY[i, :], M)  # compute idft
        y[pin - hM1:pin + hM2] += H * y1 # overlap-add to generate output sound
        pin += H                         # advance sound  pointer
        # delete half of first window which was added in stftAnalysis
        y = np.delete(y, range(hM2))
        # delete the end of the sound that was added in stftAnal
        y = np.delete(y, range(y.size-hM1, y.size)) 
        
        return y


# Test
''' # remove triple quotes to execute
import matplotlib.pyplot as plt
from scipy.signal import get_window
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '../software/models/'))
import utility_functions as UF

inputFile = '../sms-tools/sounds/piano.wav'
window = 'hamming'
M = 1024
N = 1024
H = 512

(fs, x) = UF.wavread(inputFile)

w = get_window(window, M)

mX, pX = stft_analysis(x, w, N, H)

plt.plot(x)
plt.show()

plt.plot(mX[50, :])
plt.show()

plt.plot(pX[50, :])
plt.show()

plt.pcolormesh(np.transpose(mX))
plt.show()
'''

import numpy as np
import math
from scipy.fftpack import fft, ifft
import utility_functions as UF

def dft_analysis(x, w, n, tol = 1e-14):
    '''
    Analysis of a signal using the discrete Fourier transform               
    
    @param x (list): 
      input signal
    @param w (list): analysis window
    @param n (int):
      FFT size                       
    @param tol:
      threshold used to compute phase spectrum
    
    @return (mX, pX): 
      (magnitude spectrum, phase spectrum)
    '''

    if not(UF.isPower2(n)):
        raise ValueError('FFT size (n) is not a power of 2.')

    if(w.size > n):
        raise ValueError('Window size (m) is begger than FFT size (n).')

    # Size of positive spectrum (inludes sample 0)
    hN = (n / 2) + 1

    # Half analysis window size by rounding and by flooring
    hM1 = int(math.floor((w.size + 1) / 2)) 
    hM2 = int(math.floor(w.size / 2))

    # Initialize FFT buffer
    fft_buffer = np.zeros(n)
    # Initialize output array
    y = np.zeros(x.size)

    # Window the input sound
    xw = x * w

    # Zero-phase window in buffer (Split xw in half and swap halves along
    # with any 0-padding)
    fft_buffer[:hM1] = xw[hM2:]
    fft_buffer[-hM2:] = xw[:hM2]

    # Do FFT
    X = fft(fft_buffer)

    # Compute magnitude of positive side
    absX = abs(X[:hN])
    # If 0s present add epsilon to prevent log error
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps
    # And convert to decibels
    mX = 20 * np.log10(absX)

    # Phases
    # For phase calculations zero all small values
    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0
    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0
    pX = np.unwrap(np.angle(X[:hN]))
    
    return mX, pX



def dft_synthesis(mX, pX, m):
    '''
    Synthesis of a signal using the discrete Fourier transform
              
    @param mX: 
      magnitude spectrum
    @param pX: 
      phase spectrum
    @param m: 
      window size              
    
    @return y: 
      output signal                                                
    '''
    # size of positive spectrum (includes sample 0)
    hN = mX.size
    # FFT size
    n = (hN - 1) * 2                         

    if not(UF.isPower2(n)):                                
        raise ValueError('Size of mX must be (n / 2) + 1')

    # half analysis window size by rounding
    hM1 = int(math.floor((m + 1) / 2))
    # half analysis window size by floor
    hM2 = int(math.floor(m / 2))
    
    # initialize buffer for FFT    
    fftbuffer = np.zeros(n)                                 
    # initialize output array
    y = np.zeros(m)
    # Initialize output spectrum
    Y = np.zeros(n, dtype = complex)
    
    # Generate positive frequencies (inverse dB conversion)
    Y[:hN] = 10 ** (mX / 20) * np.exp(1j * pX)
    # ...and negative frequencies
    Y[hN:] = 10 ** (mX[-2:0:-1] / 20) * np.exp(-1j * pX[-2:0:-1])

    # Inverse FFT
    fft_buffer = np.real(ifft(Y))
    # and undo zero-phase windowing
    y[:hM2] = fft_buffer[-hM2:]
    y[hM2:] = fft_buffer[:hM1]

    return y

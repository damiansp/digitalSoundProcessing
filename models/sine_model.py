import dft_model as dft
import math
import numpy as np
import utility_functions as uf
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft, fftshift



def sine_tracking(p_freq,
                  p_mag,
                  p_phase,
                  t_freq,
                  freq_dev_offset = 20,
                  freq_dev_slope = 0.01):
    '''
    Tracking sinusoids from one frame to the next                           
    p_freq, p_mag, p_phase: frequencies and magnitude of current frame         
    t_freq: frequencies of incoming tracks from previous frame               
    freq_dev_offset: minimum frequency deviation at 0Hz                       
    freq_dev_slope: slope increase of minimum frequency deviation             
    
    returns t_freq_n, t_mag_n, t_phase_n: frequency, magnitude and phase of 
    tracks                                                                  
    '''

    t_freq_n  = np.zeros(t_freq.size)
    t_mag_n   = np.zeros(t_freq.size)
    t_phase_n = np.zeros(t_freq.size)

    # indices of current peaks
    p_indices = np.array(np.nonzero(p_freq), dtype = np.int)[0]
    # indices of incoming tracks (for all samples except the first)
    incoming_tracks = np.array(np.nonzero(t_freq), dtype = np.int)[0]
    new_tracks = np.zeros(t_freq.size, dtype = np.int) - 1 # initialize to -1
    # order current peaks by magnitude
    mag_order = np.argsort(-p_mag[p_indices])
    # copy current peaks, mags, and phases to temp arrays
    p_freq_t  = np.copy(p_freq)
    p_mag_t   = np.copy(p_mag)
    p_phase_t = np.copy(p_phase)

    # continue incoming tracks
    if incoming_tracks.size > 0:
        # iterate over current peaks
        for i in mag_order:
            # break if no more incoming tracks
            if incoming_tracks.size == 0:
                break
            # closest incoming track to peak
            track = np.argmin(abs(p_freq_t[i] - t_freq[incoming_tracks]))
            freq_distance = abs(p_freq[i] - t_freq[incoming_tracks[track]])
                                                
            # if distance is within threshold
            if freq_distance < (freq_dev_offset + freq_dev_slope * p_freq[i]):
                # assign peak index to track index
                new_tracks[incoming_tracks[track]] = i
                # delete index of track in incoming tracks
                incoming_tracks = np.delete(incoming_tracks, track)

    index_t = np.array(np.nonzero(new_tracks != -1), dtype = np.int)[0]

    if index_t.size > 0:
        index_p = new_tracks[index_t]           # inds of assigned peaks
        t_freq_n[index_t]  = p_freq_t[index_p]  # output freq...
        t_mag_n[index_t]   = p_mag_t[index_p]   # ...mag
        t_phase_n[index_t] = p_phase_t[index_p] # ...and phase tracks
        
        p_freq_t  = np.delete(p_freq_t,  index_p) # delete used peaks
        p_mag_t   = np.delete(p_mag_t,   index_p) # ...
        p_phase_t = np.delete(p_phase_t, index_p) # ...

    # Create new tracks from unused peaks
    # index of empty incoming tracks
    empty_t = np.array(np.nonzero(t_freq == 0), dtype = np.int)[0]
    # sort remaining peaks by mag
    peaks_left = np.argsort(-p_mag_t)

    if ((peaks_left.size > 0) and (empty_t.size >= peaks_left.size)):
        # Fill empty tracks
        t_freq_n[empty_t[:peaks_left.size]]  = p_freq_t[peaks_left]
        t_mag_n[empty_t[:peaks_left.size]]   = p_mag_t[peaks_left]
        t_phase_n[empty_t[:peaks_left.size]] = p_phase_t[peaks_left]
    elif ((peaks_left.size > 0) and (empty_t.size < peaks_left.size)):
        t_freq_n[empty_t] = p_freq_t[peaks_left[:empty_t.size]]
        t_mag_n[empty_t] = p_mag_t[peaks_left[:empty_t.size]]
        t_phase_n[empty_t] = p_phase_t[peaks_left[:empty_t.size]]
        t_freq_n = np.append(t_freq_n, p_freq_t[peaks_left[empty_t.size:]])
        t_mag_n = np.append(t_mag_n, p_mag_t[peaks_left[empty_t.size:]])
        t_phase_n = np.append(t_phase_n, p_phase_t[peaks_left[empty_t.size:]])

    return t_freq_n, t_mag_n, t_phase_n
                                                            



def clean_sine_tracks(t_freq, min_track_length = 3):
    '''
    Delete short fragments of a collection of sinusoidal tracks             
    t_freq: frequency of tracks                                              
    min_track_length: minimum duration of tracks in number of frames          
    
    returns t_freq_n: output frequency of tracks                              
    '''

    if t_freq.shape[1] == 0:
        return t_freq

    n_frames = t_freq[:, 0].size
    n_tracks = t_freq[0, :].size # number of tracks in a frame

    # iterate over all tracks
    for t in range(n_tracks):
        track_freqs = t_freq[:, t] # frequencies of one track
        # begining of track contours
        track_begs = np.nonzero((track_freqs[:n_frames - 1] <= 0) &
                                (track_freqs[1:] > 0))[0] + 1
        if track_freqs[0] > 0:
            track_begs = np.insert(track_begs, 0, 0)

        # end of track contours
        track_ends = np.nonzero((track_freqs[:n_frames - 1] > 0) &
                                (track_freqs[1:] <= 0))[0] + 1
        if track_freqs[n_frames - 1] > 0:
            track_ends = np.append(track_ends, n_frames - 1)
                                
        # lengths of trach contours
        track_lengths = 1 + track_ends - track_begs

        for i, j in zip(track_begs, track_lengths):
            # delete short track contours
            if j <= min_track_length:
                track_freqs[i:i + j] = 0

    return t_freq



def sine_model_analysis(x,
                        fs,
                        w,
                        n,
                        h,
                        t,
                        max_n_sines = 100,
                        min_sine_duration = 0.01,
                        freq_dev_offeset = 20,
                        freq_dev_slope = 0.01):
    '''
    Analysis of a sound using the sinusoidal model with sine tracking       
    x: input array sound
    w: analysis window
    n: size of complex spectrum
    h: hop-size
    t: threshold in negative dB                                        
    max_n_sines: maximum number of sines per frame
    min_sine_duration: minimum duration of sines in seconds
    freq_dev_offset: minimum frequency deviation at 0 Hz
    freq_dev_slope: slope increase of minimum frequency deviation (needed to 
        adjust for the fact that higher freqs tend to have higher deviation)
    
    returns xtfreq, xtmag, xtphase: frequencies, magnitudes and phases of 
        sinusoidal tracks
    '''

    # raise error if min_sine_duration is smaller than 0
    if (minSineDur <0):    
        raise ValueError("Minimum duration of sine tracks smaller than 0")

    hM1 = int(np.floor((w.size + 1) / 2))
    hM2 = int(np.floor(w.size / 2))
    x = np.append(np.zeros(hM2), x)
    x = np.append(x,np.zeros(hM2))

    # initialize sound pointer at middle of window
    pin = hM1
    pend = x.size - hM1
    # normalize window
    w = w / sum(w)

    while pin < pend:
        x1 = x[pin - hM1:pin + hM2]
        mX, pX = dft.dft_analysis(x1, w, n)
        p_loc = uf.peakDetection(mX, t)
        ip_loc, ip_mag, ip_phase = uf.peakInterp(mX, pX, p_loc)
        ip_freq = fs * ip_loc / float(n)

        # Sine-Tracking
        t_freq, t_mag, t_phase = sine_tracking(
            ip_freq, ip_mag, ip_phase, t_freq, freq_dev_offset, freq_dev_slope)

        # limit no. of tracks to max_n_sines
        t_freq  = np.resize(t_freq,  min(max_n_sines, t_freq.size))
        t_mag   = np.resize(t_mag,   min(max_n_sines, t_mag.size))
        t_phase = np.resize(t_phase, min(max_n_sines, t_phase.size))

        # Temp output arrays:
        jt_freq  = np.zeros(max_n_sines)
        jt_mag   = np.zeros(max_n_sines)
        jt_phase = np.zeros(max_n_sines)

        # Save tracks to temp arrays:
        jt_freq[:t_freq.size]   = t_freq
        jt_mag[:t_mag.size]     = t_mag
        jt_phase[:t_phase.size] = t_phase

        # if first frame, init output sine tracks
        if pin == hM1:
            xt_freq  = jt_freq
            xt_mag   = jt_mag
            xt_phase = jt_phase
        else:
            xt_freq  = np.vstack((xt_freq,  jt_freq))
            xt_mag   = np.vstack((xt_mag,   jt_mag))
            xt_phase = np.vstack((xt_phase, jt_phase))

        # hop to next frame
        pin += h

    # delete sine tracks shorter than minSineDur
    xt_freq = clean_sine_tracks(xt_freq, round(fs * min_sine_duration / h))

    return xt_freq, xt_mag, xt_phase




def sine_model_synthesis(t_freq, t_mag, t_phase, n, h, fs):
    '''
    Synthesis of a sound using the sinusoidal model                         
    t_freq, t_mag, t_phase: frequencies, magnitudes and phases of sinusoids 
    n: synthesis FFT size
    h: hop size
    fs: sampling rate                   

    returns y: output array sound                                           
    '''
    h_n = n / 2 # half of FFT for synth
    L = t_freq.shape[0] # no. frames
    p_out = 0           # init ouputput sound pointer
    y_size = h * (L + 3) # output size
    y = np.zeros(y_size) # init output array
    sw = np.zeros(n)     # init synth window
    tw = triang(2 * h)   # triangular window
    sw[h_n - h : h_n + h] = tw # add triang window
    bh = blackmanharris(n_s) # blackman-harris window

    # normalize windows
    bh = bh / sum(bh)
    sw[h_n - h : h_n + h] = sw[h_n - h : h_n + h] / bh[h_n - h : h_n + h]

    last_y_t_freq = t_freq[0, :]                        # init synth freqs...
    y_t_phase = 2 * np.pi * np.random.rand(t_freq[0, :].size) # ...and phases

    
                
    # For all frames
    for l in range(L):
        # if no phases, then generate
        if (t_phase.size > 0):
            y_t_phase = t_phase[l, :]
        else:
            y_t_phase += (np.pi * (last_y_t_freq + t_freq[l, :]) / fs) * h
            
        # generate sines in spectrum
        Y = uf.genSpecSines(t_freq[l, :], t_mag[l, :], y_t_phase, n_s, fs)
        # save freq for phase propagation
        last_y_freq = t_freq[l, :]
        # bound freq inside 2pi
        y_t_phase = y_t_phase % (2 * np.pi)
        # Inverse FFT
        yw = np.real(fftshift(ifft(Y)))

        # Overlap-add and apply synth window
        y[p_out : p_out + n_s] += sw * yw
        p_out += h

    # delete half of first window...
    y = np.delete(y, range(h_n))
    # ...and half of last window
    y = np.delete(y, range(y.size - h_n, y.size))

    return y

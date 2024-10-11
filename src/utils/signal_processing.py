import numpy as np
import matplotlib.pyplot as plt
import pywt


def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')
  
def filter_by_psd(data, dt, timestamps, cutoff=100, visualize=True, title="", y_label=""):
  """
    Filter signal by power spectrum density
  """
  n = data.shape[0] 
  fhat = np.fft.fft(data, n) # Compute the FFT
  PSD = fhat * np.conj(fhat) / n    # Power spectrum (power per frequency)
  freq = (1/(dt * n)) * np.arange(n) # Create x-axis of frequencies
  L = np.arange(1, np.floor(n/2), dtype="int") # Only plot the first half of frequencies
  
  indices = PSD > cutoff
  PSD_clean = PSD * indices
  fhat_filtered = indices * fhat
  ffilt = np.fft.ifft(fhat_filtered).real

  if visualize:
    fig, axs = plt.subplots(2, 1)
    plt.sca(axs[0])
    plt.title(title)
    plt.plot(timestamps, data, color='b', lw=1, label='')
    plt.plot(timestamps, ffilt, color='red', lw=1, label='')
    plt.xlabel("Timestamp(ns)")
    plt.ylabel(y_label)
    plt.grid()
    
    plt.sca(axs[1])
    plt.title("Frequency vs Power spectrum")
    plt.plot(freq[L], PSD[L], color='b', lw=1, label='Before filtering')
    plt.plot(freq[L], PSD_clean[L], color='red', lw=1, label='After filtering')
    plt.xlabel("Frequencies (Hz)")
    plt.ylabel("Power spectrum density")
    plt.legend()
    plt.grid()
    fig.tight_layout()
    
    plt.show()

  return ffilt
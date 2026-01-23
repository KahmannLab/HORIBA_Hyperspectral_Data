import numpy as np
import matplotlib.pyplot as plt
#%% Remove spikes from 3d dataset with numba for speed up by parallelization
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d

def spike_removal_1d(
    y,
    width_threshold,
    prominence_threshold=None,
    window=10,
    rel_height=0.8,
    interp_type="linear",
):
    """
    Remove narrow spikes from a 1D signal by peak detection and interpolation.

    Parameters
    ----------
    y : array_like
        Input 1D signal
    width_threshold : float
        Maximum peak width considered a spike
    prominence_threshold : float or None
        Prominence threshold for peak detection
    window : int
        Half-width of local interpolation window
    rel_height : float
        Relative height used to estimate peak extent
    interp_type : str
        Interpolation type passed to scipy.interpolate.interp1d

    Returns
    -------
    y_out : ndarray
        Despiked signal
    """
    y = np.asarray(y)
    N = y.size
    y_out = y.copy()

    # --- Detect peaks ---
    peaks, _ = find_peaks(y, prominence=prominence_threshold)
    if len(peaks) == 0:
        return y_out

    widths, _, left_ips, right_ips = peak_widths(y, peaks, rel_height=rel_height)

    # --- Mark spike regions ---
    spikes = np.zeros(N, dtype=bool)

    for w, a, b in zip(widths, left_ips, right_ips):
        if w < width_threshold:
            lo = max(int(a) - 1, 0)
            hi = min(int(b) + 1, N - 1)
            spikes[lo : hi + 1] = True

    # --- Interpolate over spikes ---
    for i in np.where(spikes)[0]:
        lo = max(0, i - window)
        hi = min(N, i + window + 1)

        neighbors = np.arange(lo, hi)
        neighbors = neighbors[~spikes[neighbors]]

        if neighbors.size < 2:
            continue

        interp = interp1d(
            neighbors,
            y[neighbors],
            kind=interp_type,
            bounds_error=False,
            fill_value="extrapolate",
        )

        y_out[i] = interp(np.clip(i, neighbors.min(), neighbors.max()))

    return y_out

from joblib import Parallel, delayed

def spike_removal_3d(
    cube,
    width_threshold,
    prominence_threshold=None,
    window=10,
    rel_height=0.8,
    interp_type="linear",
    n_jobs=-1,
):
    """
    Apply 1D spike removal along the last axis of a 3D cube.
    """
    X, Y, _ = cube.shape
    cube_out = np.empty_like(cube)

    def process(ix, iy):
        return spike_removal_1d(
            cube[ix, iy],
            width_threshold,
            prominence_threshold,
            window,
            rel_height,
            interp_type,
        )

    results = Parallel(n_jobs=n_jobs)(
        delayed(process)(ix, iy)
        for ix in range(X)
        for iy in range(Y)
    )

    k = 0
    for ix in range(X):
        for iy in range(Y):
            cube_out[ix, iy] = results[k]
            k += 1

    return cube_out
#%% FFT filtering for oscillating pattern removal
from scipy.signal import find_peaks

def plot_avg_fft_with_noise_peaks(
    cube,
    noise_freq_start=50,
    prominence=5.0,
    distance=3,
    log_scale=True,
    xlim=None,
):
    """
    Plot averaged FFT magnitude spectrum over all pixels
    and mark peaks in the noise frequency regime.

    Parameters
    ----------
    cube : ndarray (X, Y, N)
        Hyperspectral data cube
    noise_freq_start : int
        FFT index above which frequencies are considered noise-dominated
    prominence : float
        Peak prominence for peak detection (relative to local background)
    distance : int
        Minimum distance between detected peaks (in FFT bins)
    log_scale : bool
        Use log-scale for magnitude
    xlim : tuple or None
        Frequency index limits for zoom-in
    """
    X, Y, N = cube.shape

    # --- FFT (real-valued) ---
    Yf = np.fft.rfft(cube.astype(np.float32), axis=-1)
    mag = np.abs(Yf)

    # --- Average over pixels ---
    mag_avg = mag.mean(axis=(0, 1))
    freqs = np.arange(mag_avg.size)

    # --- Detect peaks in noise regime ---
    noise_mask = freqs >= noise_freq_start
    peaks, props = find_peaks(
        mag_avg[noise_mask],
        prominence=prominence,
        distance=distance,
    )

    # Convert to full-spectrum indices
    noise_peaks = peaks + noise_freq_start

    # --- Plot ---
    plt.figure()
    plt.plot(freqs, mag_avg, label="Averaged FFT magnitude")

    if noise_peaks.size > 0:
        plt.scatter(
            noise_peaks,
            mag_avg[noise_peaks],
            color="r",
            s=40,
            label="Detected noise peaks",
            zorder=3,
        )
        # --- Label peaks ---
        for p in noise_peaks:
            plt.annotate(
                f"{p}",
                (p, mag_avg[p]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="r",
            )

    plt.axvline(
        noise_freq_start,
        color="gray",
        linestyle="--",
        label="Noise regime start",
    )

    plt.xlabel("FFT frequency index")
    plt.ylabel("Average magnitude")
    plt.title("Averaged FFT spectrum with noise-peak detection")

    if log_scale:
        plt.yscale("log")
    if xlim is not None:
        plt.xlim(xlim)

    plt.legend()
    plt.tight_layout()
    plt.show()

    return noise_peaks

def fft_notch_filter_1d(
    y,
    notch_freqs,
    notch_width=1):
    """
    Remove oscillatory components from a 1D signal using FFT notch filtering.

    Parameters
    ----------
    y : array_like
        Input spectrum
    notch_freqs : list or array
        Indices of FFT frequencies to suppress (positive frequencies)
    notch_width : int
        Half-width of each notch (in frequency bins)

    Returns
    -------
    y_out : ndarray
        Filtered spectrum
    """
    y = np.asarray(y, dtype=float)
    N = y.size

    # FFT
    Y = np.fft.fft(y)

    for f in notch_freqs:
        f = int(f)
        lo = max(f - notch_width, 0)
        hi = min(f + notch_width + 1, N)

        # Remove positive and symmetric negative frequencies
        Y[lo:hi] = 0
        Y[-hi:-lo] = 0

    # Inverse FFT
    y_out = np.real(np.fft.ifft(Y))

    return y_out

def fft_notch_filter_3d(
    cube,
    notch_freqs,
    notch_width=1,
    n_jobs=-1):
    """
    Apply FFT notch filtering along the spectral axis of a 3D hyperspectral cube.
    """
    X, Y, _ = cube.shape
    cube_out = np.empty_like(cube, dtype=float)

    def process(ix, iy):
        return fft_notch_filter_1d(
            cube[ix, iy],
            notch_freqs=notch_freqs,
            notch_width=notch_width,
        )

    results = Parallel(n_jobs=n_jobs)(
        delayed(process)(ix, iy)
        for ix in range(X)
        for iy in range(Y)
    )

    k = 0
    for ix in range(X):
        for iy in range(Y):
            cube_out[ix, iy] = results[k]
            k += 1

    return cube_out
#%% Moving average smoother for Gaussian noise removal
def moving_average_1d(y, window=7):
    """
    Moving average (mean) smoother for 1D spectra.

    Parameters
    ----------
    y : array_like
        Input spectrum
    window : int
        Window size (odd recommended)

    Returns
    -------
    y_out : ndarray
        Smoothed spectrum
    """
    y = np.asarray(y, dtype=float)

    if window < 1:
        raise ValueError("window must be >= 1")

    if window % 2 == 0:
        raise ValueError("window size should be odd")

    # pad the signal to handle edge effects
    y_pad = np.pad(y, window // 2, mode="reflect")
    kernel = np.ones(window) / window
    y_out = np.convolve(y_pad, kernel, mode="valid")

    return y_out
def moving_average_3d_parallel(
    cube,
    window=7,
    n_jobs=-1,
):
    """
    Parallel moving average smoothing for hyperspectral data.
    """
    X, Y, _ = cube.shape
    cube_out = np.empty_like(cube, dtype=float)

    def process(ix, iy):
        return moving_average_1d(
            cube[ix, iy], window=window
        )

    results = Parallel(n_jobs=n_jobs)(
        delayed(process)(ix, iy)
        for ix in range(X)
        for iy in range(Y)
    )

    k = 0
    for ix in range(X):
        for iy in range(Y):
            cube_out[ix, iy] = results[k]
            k += 1

    return cube_out

#%% Baseline correction via Asymmetric least square
from pybaselines import Baseline

def als_baseline_1d(spectrum, lam=1e5, p=0.01):
    """
    Apply ALS baseline to a single spectrum.
    """
    bl = Baseline()
    baseline, _ = bl.asls(spectrum, lam=lam, p=p)
    return baseline

def pixelwise_als_baseline(
    cube,
    lam=1e5,
    p=0.01,
    n_jobs=-1,
    return_baseline=False,
):
    """
    Pixel-wise ALS baseline subtraction for Raman hyperspectral cube.

    Parameters
    ----------
    cube : ndarray (nx, ny, n_wavenumber)
        Raman hyperspectral data
    lam : float
        ALS smoothness parameter
    p : float
        ALS asymmetry parameter
    n_jobs : int
        Number of parallel jobs (-1 = all cores)
    return_baseline : bool
        If True, also return baseline cube

    Returns
    -------
    corrected_cube : ndarray
    baseline_cube : ndarray (optional)
    """

    nx, ny, n_spec = cube.shape
    spectra = cube.reshape(-1, n_spec)

    # Parallel ALS
    baselines = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(als_baseline_1d)(spec, lam, p)
        for spec in spectra
    )

    baseline_cube = np.array(baselines).reshape(nx, ny, n_spec)
    corrected_cube = cube - baseline_cube

    if return_baseline:
        return corrected_cube, baseline_cube
    else:
        return corrected_cube

#%% Adaptive median filter for despiking and denoising
def adaptive_median_filter_1d(
    y,
    window=7,
    spike_threshold=3.5,
    smooth_weight=0.5,
):
    """
    Adaptive median filter that handles both impulse noise (spikes)
    and Gaussian noise in 1D spectra.

    Parameters
    ----------
    y : array_like
        Input 1D signal (spectrum)
    window : int
        Odd window size
    spike_threshold : float
        Threshold (in MAD units) to classify a point as a spike
    smooth_weight : float
        Weight between median and mean for non-spike points (0–1)

    Returns
    -------
    y_out : ndarray
        Filtered signal
    """
    y = np.asarray(y, dtype=float)
    N = y.size
    half = window // 2
    y_out = y.copy()

    for i in range(N):
        lo = max(0, i - half)
        hi = min(N, i + half + 1)
        local = y[lo:hi]

        median = np.median(local)
        mad = np.median(np.abs(local - median)) + 1e-12
        mean = local.mean()

        # Spike detection using robust z-score
        z = np.abs(y[i] - median) / mad

        if z > spike_threshold:
            # Impulse noise → median replacement
            y_out[i] = median
        else:
            # Gaussian noise → mild smoothing
            y_out[i] = (
                smooth_weight * median
                + (1 - smooth_weight) * mean
            )

    return y_out


def adaptive_median_filter_3d_parallel(
    cube,
    n_jobs=-1,
    **kwargs,
):
    """
    Parallel adaptive median filtering along spectral axis of a 3D cube.
    """
    X, Y, _ = cube.shape
    cube_out = np.empty_like(cube)

    def process(ix, iy):
        return adaptive_median_filter_1d(cube[ix, iy], **kwargs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(process)(ix, iy)
        for ix in range(X)
        for iy in range(Y)
    )

    k = 0
    for ix in range(X):
        for iy in range(Y):
            cube_out[ix, iy] = results[k]
            k += 1

    return cube_out
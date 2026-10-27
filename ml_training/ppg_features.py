"""
Python twin of the Android feature extractor (PPGFeatureExtractor.java / PpgDsp.java).

FEATURE_VERSION must match the Java constant. Any change here must be made in Java too, followed by
regenerating the golden vectors (python make_golden_vectors.py) and re-running the Java parity test.
"""
import numpy as np

FEATURE_VERSION = "ppgfeat-1"
TARGET_FS = 30.0
MIN_DURATION_SEC = 10.0

CARDIAC_F0, CARDIAC_STEP, CARDIAC_BINS = 0.60, 0.01, 301
RESP_F0, RESP_STEP, RESP_BINS = 0.08, 0.01, 53
PEAK_HALF_WINDOW = 7

FEATURE_NAMES = [
    "g_dom_freq_hz", "g_peak_power_ratio", "g_second_peak_ratio", "g_spectral_entropy",
    "r_dom_freq_hz", "r_peak_power_ratio", "g_harmonic_ratio",
    "g_ibi_mean_s", "g_ibi_std_s", "g_rmssd_s", "g_peak_count", "g_hr_from_peaks_bpm",
    "g_ac_std", "g_ac_dc_ratio", "g_skewness", "g_excess_kurtosis",
    "g_zero_cross_rate_hz", "g_peak_amp_mean",
    "resp_g_dom_freq_hz", "resp_g_peak_power_ratio", "resp_r_dom_freq_hz",
    "red_ac_dc_ratio", "blue_ac_dc_ratio", "ratio_of_ratios", "red_dc_norm", "duration_s",
]
DEMOGRAPHIC_NAMES = ["demo_age", "demo_height_cm", "demo_weight_kg", "demo_gender"]


class FeatureExtractionError(ValueError):
    pass


# ---------------------------------------------------------------- DSP primitives (PpgDsp.java)
def resample_linear(t, x, fs):
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    tt = t - t[0]
    total = tt[-1]
    m = int(np.floor(total * fs + 1e-9)) + 1
    grid = np.arange(m, dtype=np.float64) / fs
    return np.interp(grid, tt, x)


def moving_average(x, w):
    n = len(x)
    h = w // 2
    prefix = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - h)
    hi = np.minimum(n - 1, idx + h)
    return (prefix[hi + 1] - prefix[lo]) / (hi - lo + 1)


def linear_detrend(x):
    n = len(x)
    mi = (n - 1) / 2.0
    mx = x.mean()
    i = np.arange(n)
    sxy = np.sum((i - mi) * (x - mx))
    sxx = np.sum((i - mi) ** 2)
    slope = 0.0 if sxx == 0 else sxy / sxx
    return x - (mx + slope * (i - mi))


def cardiac_band(x):
    return moving_average(x - moving_average(x, 45), 5)


def hann(x):
    n = len(x)
    if n > 1:
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
    else:
        w = np.ones(n)
    return x * w


def power_at(s, fs, f):
    ang = 2.0 * np.pi * f / fs * np.arange(len(s))
    re = np.sum(s * np.cos(ang))
    im = -np.sum(s * np.sin(ang))
    return re * re + im * im


def spectrum(x, fs, f0, step, count):
    s = hann(x)
    return np.array([power_at(s, fs, f0 + j * step) for j in range(count)])


def _cardiac_spectrum(cardiac):
    p = spectrum(cardiac, TARGET_FS, CARDIAC_F0, CARDIAC_STEP, CARDIAC_BINS)
    k = int(np.argmax(p))
    total = float(p.sum())
    near = float(p[max(0, k - 10): min(CARDIAC_BINS - 1, k + 10) + 1].sum())
    return p, k, CARDIAC_F0 + k * CARDIAC_STEP, total, (near / total if total > 0 else 0.0)


def _condition(t, r, g, b):
    t = np.asarray(t, dtype=np.float64)
    arrs = [np.asarray(a, dtype=np.float64) for a in (r, g, b)]
    if len(t) < 2:
        raise FeatureExtractionError("fewer than two samples")
    if not (np.all(np.isfinite(t)) and all(np.all(np.isfinite(a)) for a in arrs)):
        raise FeatureExtractionError("non-finite values")
    if np.any(np.diff(t) <= 0):
        raise FeatureExtractionError("timestamps not strictly increasing")
    duration = float(t[-1] - t[0])
    if duration < MIN_DURATION_SEC:
        raise FeatureExtractionError("recording shorter than %d s" % MIN_DURATION_SEC)
    R, G, B = (resample_linear(t, a, TARGET_FS) for a in arrs)
    return R, G, B, duration


def extract_features(t, red, green, blue):
    """Returns a float64 vector of len(FEATURE_NAMES) -- identical to PPGFeatureExtractor.extract()."""
    R, G, B, duration = _condition(t, red, green, blue)
    fs = TARGET_FS
    cg, cr, cb = cardiac_band(G), cardiac_band(R), cardiac_band(B)
    m = len(cg)
    f = np.zeros(len(FEATURE_NAMES))

    pg, kg, fg, totg, ratg = _cardiac_spectrum(cg)
    _, _, fr, _, ratr = _cardiac_spectrum(cr)
    f[0], f[1] = fg, ratg
    far = np.abs(np.arange(CARDIAC_BINS) - kg) > 15
    second = float(pg[far].max()) if far.any() else 0.0
    peak_power = float(pg[kg])
    f[2] = second / peak_power if peak_power > 0 else 0.0
    ent = 0.0
    if totg > 0:
        p = pg / totg
        p = p[p > 0]
        ent = float(-(p * np.log(p)).sum() / np.log(CARDIAC_BINS))
    f[3] = ent
    f[4], f[5] = fr, ratr
    harm = power_at(hann(cg), fs, 2.0 * fg)
    f[6] = harm / peak_power if peak_power > 0 else 0.0

    peaks = []
    for i in range(PEAK_HALF_WINDOW, m - PEAK_HALF_WINDOW):
        if not cg[i] > 0:
            continue
        ok = True
        for k in range(1, PEAK_HALF_WINDOW + 1):
            if not (cg[i] > cg[i - k] and cg[i] >= cg[i + k]):
                ok = False
                break
        if ok:
            peaks.append(i)
    peaks = np.array(peaks, dtype=int)
    ibi = np.diff(peaks) / fs if len(peaks) > 1 else np.zeros(0)
    ibi_mean = float(ibi.mean()) if len(ibi) else 0.0
    f[7] = ibi_mean
    f[8] = float(ibi.std()) if len(ibi) else 0.0
    f[9] = float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if len(ibi) >= 2 else 0.0
    f[10] = len(peaks)
    f[11] = 60.0 / ibi_mean if ibi_mean > 0 else 0.0

    ac_std = float(cg.std())
    dc_green = float(G.mean())
    f[12] = ac_std
    f[13] = ac_std / dc_green if dc_green > 0 else 0.0
    d = cg - cg.mean()
    m2, m3, m4 = float(np.mean(d ** 2)), float(np.mean(d ** 3)), float(np.mean(d ** 4))
    f[14] = m3 / m2 ** 1.5 if m2 > 1e-12 else 0.0
    f[15] = m4 / (m2 * m2) - 3.0 if m2 > 1e-12 else 0.0
    zc = int(np.sum(cg[:-1] * cg[1:] < 0))
    f[16] = zc / ((m - 1) / fs)
    f[17] = float(cg[peaks].mean()) if len(peaks) else 0.0

    respg = moving_average(linear_detrend(G), 15)
    respr = moving_average(linear_detrend(R), 15)
    pgr = spectrum(respg, fs, RESP_F0, RESP_STEP, RESP_BINS)
    prr = spectrum(respr, fs, RESP_F0, RESP_STEP, RESP_BINS)
    ig, ir = int(np.argmax(pgr)), int(np.argmax(prr))
    f[18] = RESP_F0 + ig * RESP_STEP
    tot = float(pgr.sum())
    near = float(pgr[max(0, ig - 5): min(RESP_BINS - 1, ig + 5) + 1].sum())
    f[19] = near / tot if tot > 0 else 0.0
    f[20] = RESP_F0 + ir * RESP_STEP

    dc_red, dc_blue = float(R.mean()), float(B.mean())
    red_acdc = float(cr.std()) / dc_red if dc_red > 0 else 0.0
    blue_acdc = float(cb.std()) / dc_blue if dc_blue > 0 else 0.0
    f[21], f[22] = red_acdc, blue_acdc
    f[23] = red_acdc / blue_acdc if blue_acdc > 1e-12 else 0.0
    f[24] = dc_red / 255.0
    f[25] = duration
    if not np.all(np.isfinite(f)):
        raise FeatureExtractionError("non-finite feature")
    return f


def demographic_vector(age, height_cm, weight_kg, gender):
    return np.array([age, height_cm, weight_kg, gender], dtype=np.float64)

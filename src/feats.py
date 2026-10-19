"""Feature extraction for rāga experiments (operates on tonic-normalised cent contours)."""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

UNV = -32768
FPS = 22.67  # ~ 44100/128/ (step 10) -> frames per second of compact contour


def voiced(c):
    return c[c != UNV].astype(np.float64)


def pitch_correct(c, alpha):
    """Simulated studio pitch correction: pull each voiced frame toward the nearest
    equal-tempered semitone (relative to tonic) by strength alpha in [0,1]."""
    if alpha <= 0:
        return c
    out = c.astype(np.float64).copy()
    m = c != UNV
    v = out[m]
    out[m] = v + alpha * (np.round(v / 100.0) * 100.0 - v)
    out = np.round(out)
    out[~m] = UNV
    return out.astype(np.int32)


def folded_hist(c, nbins):
    """Octave-folded pitch distribution with bins centred on ET positions."""
    v = voiced(c)
    w = 1200.0 / nbins
    idx = np.floor(np.mod(v + w / 2.0, 1200.0) / w).astype(int) % nbins
    h = np.bincount(idx, minlength=nbins).astype(np.float64)
    return h / max(h.sum(), 1)


def pd_feature(c, nbins, smooth_cents=0):
    h = folded_hist(c, nbins)
    if smooth_cents:
        h = gaussian_filter1d(h, smooth_cents / (1200.0 / nbins), mode="wrap")
        h /= h.sum()
    return np.sqrt(h)  # Hellinger embedding


def tdms(c, nbins=120, tau_s=0.3, sigma=2.0, alpha=0.75):
    """Time-delayed melody surface (after Gulati et al., 2016)."""
    tau = max(1, int(round(tau_s * FPS)))
    x = c[:-tau]; y = c[tau:]
    m = (x != UNV) & (y != UNV)
    w = 1200.0 / nbins
    bx = (np.floor(np.mod(x[m] + w / 2, 1200) / w).astype(int)) % nbins
    by = (np.floor(np.mod(y[m] + w / 2, 1200) / w).astype(int)) % nbins
    S = np.zeros((nbins, nbins))
    np.add.at(S, (bx, by), 1)
    S = gaussian_filter(S, sigma, mode="wrap")
    S = S ** alpha
    S /= max(S.sum(), 1e-12)
    return S.ravel()


def stable_segments(c, thr=25.0, min_len=4):
    """Return (start, end) of stable-note segments: consecutive voiced frames
    whose pitch stays within +-thr cents of the segment median."""
    segs = []
    n = len(c); i = 0
    while i < n:
        if c[i] == UNV:
            i += 1; continue
        j = i; ref = c[i]
        while j + 1 < n and c[j + 1] != UNV and abs(int(c[j + 1]) - int(ref)) <= thr:
            j += 1
        if j - i + 1 >= min_len:
            segs.append((i, j + 1))
        i = j + 1
    return segs


def shruti_descriptors(c):
    """Per-svara microtonal descriptors (12 svaras x 7) = 84 dims.
    mass, peak offset from ET, intonation spread, skew, stable-dwell fraction,
    mean dwell duration, gamaka (oscillation) energy."""
    v = voiced(c)
    fold = np.mod(v + 50.0, 1200.0) - 50.0  # in [-50, 1150)
    sv = np.clip(np.round(fold / 100.0).astype(int), 0, 11)
    dev = fold - sv * 100.0  # deviation from ET in [-50, 50)
    h1 = np.bincount(np.mod(np.round(v).astype(int), 1200), minlength=1200).astype(float)
    h1 = gaussian_filter1d(h1, 3, mode="wrap")
    # contour derivatives for gamaka
    cv = c.astype(np.float64); mv = c != UNV
    d = np.zeros_like(cv); both = mv[1:] & mv[:-1]
    d[1:][both] = np.abs(np.diff(cv))[both]
    vfold = np.full(len(c), -1)
    vfold[mv] = np.clip(np.round((np.mod(cv[mv] + 50.0, 1200.0) - 50.0) / 100.0).astype(int), 0, 11)
    segs = stable_segments(c)
    seg_sv = [];
    seg_len = []
    for a, b in segs:
        med = np.median(cv[a:b])
        seg_sv.append(int(np.clip(np.round((np.mod(med + 50, 1200) - 50) / 100), 0, 11)))
        seg_len.append((b - a) / FPS)
    seg_sv = np.array(seg_sv, int); seg_len = np.array(seg_len)
    total_voiced = max(mv.sum(), 1)
    F = []
    for s in range(12):
        m = sv == s
        mass = m.mean() if len(sv) else 0
        if m.sum() > 10:
            region = np.arange(s * 100 - 50, s * 100 + 50) % 1200
            pk = region[np.argmax(h1[region])]
            off = ((pk - s * 100 + 600) % 1200) - 600
            dd = dev[m]; spread = dd.std(); skew = ((dd - dd.mean()) ** 3).mean() / (spread ** 3 + 1e-9)
        else:
            off = spread = skew = 0.0
        ms = seg_sv == s
        dwell_frac = seg_len[ms].sum() * FPS / total_voiced if ms.any() else 0.0
        dwell_mean = seg_len[ms].mean() if ms.any() else 0.0
        gm = vfold == s
        gam = np.median(d[gm]) if gm.sum() > 10 else 0.0
        F += [mass, off / 50.0, spread / 50.0, np.clip(skew, -5, 5) / 5, dwell_frac, np.log1p(dwell_mean), np.log1p(gam)]
    return np.array(F, dtype=np.float64)


def microtonal_deviation(c):
    """Mean absolute deviation (cents) of stable-note medians from ET grid."""
    vals = []
    for a, b in stable_segments(c):
        med = np.median(c[a:b].astype(float))
        vals.append(abs(med - np.round(med / 100) * 100))
    return float(np.mean(vals)) if vals else np.nan


def excerpt_roll(c, start, L, nbins=120, sigma=1.0):
    """Folded pitch 'piano-roll' (nbins x L) for the deep model."""
    seg = c[start:start + L]
    R = np.zeros((nbins, L), np.float32)
    m = seg != UNV
    if m.any():
        w = 1200.0 / nbins
        b = (np.floor(np.mod(seg[m] + w / 2, 1200) / w).astype(int)) % nbins
        R[b, np.nonzero(m)[0]] = 1.0
        if sigma:
            R = gaussian_filter1d(R, sigma, axis=0, mode="wrap")
    return R

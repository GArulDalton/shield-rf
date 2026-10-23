import numpy as np

N = 128                     # preamble length (samples)
FS = 1.0                    # normalized sampling rate

def zc(n=N, u=25):
    k = np.arange(n)
    return np.exp(-1j * np.pi * u * k * (k + 1) / n)

PRE = zc()

def make_devices(n_dev, seed=0):
    rng = np.random.default_rng(seed)
    devs = []
    for _ in range(n_dev):
        devs.append(dict(
            eps=rng.uniform(-0.08, 0.08),          # IQ amplitude imbalance
            phi=rng.uniform(-6, 6) * np.pi / 180,  # IQ phase imbalance (rad)
            cfo=rng.uniform(-4e-3, 4e-3),          # normalized CFO
            dc=(rng.normal(0, 0.03) + 1j * rng.normal(0, 0.03)),
            a3=rng.uniform(0.0, 0.12) * np.exp(1j * rng.uniform(-0.5, 0.5)),  # PA 3rd-order
            pn=rng.uniform(0.002, 0.02),           # phase-noise std per sample (rad)
        ))
    return devs

def tx(dev, rng):
    s = PRE.copy()
    # PA nonlinearity (memoryless polynomial)
    s = s - dev['a3'] * s * np.abs(s) ** 2
    # IQ imbalance
    mu = np.cos(dev['phi']) + 1j * dev['eps'] * np.sin(dev['phi'])
    nu = dev['eps'] * np.cos(dev['phi']) - 1j * np.sin(dev['phi'])
    s = mu * s + nu * np.conj(s)
    s = s + dev['dc']
    # CFO (+ small per-burst drift) and phase noise
    n = np.arange(N)
    cfo = dev['cfo'] * (1 + rng.normal(0, 0.02))
    ph = 2 * np.pi * cfo * n + np.cumsum(rng.normal(0, dev['pn'], N)) + rng.uniform(0, 2 * np.pi)
    return s * np.exp(1j * ph)

# channel profiles: name -> generator of impulse response + doppler
CHANNELS = ['AWGN', 'Rician', 'TDL-A', 'Rayleigh', 'TDL-C']

def channel(x, name, rng):
    if name == 'AWGN':
        h = np.array([1.0 + 0j])
    elif name == 'Rician':
        K = 10 ** (rng.uniform(3, 10) / 10)
        los = np.sqrt(K / (K + 1)) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        h = np.array([los + np.sqrt(1 / (K + 1)) * (rng.normal() + 1j * rng.normal()) / np.sqrt(2),
                      0.3 * (rng.normal() + 1j * rng.normal()) / np.sqrt(2)])
    elif name == 'TDL-A':   # short delay spread
        pdp = np.array([1.0, 0.5, 0.2])
        h = np.sqrt(pdp / pdp.sum()) * (rng.normal(size=3) + 1j * rng.normal(size=3)) / np.sqrt(2)
    elif name == 'Rayleigh':
        h = np.array([(rng.normal() + 1j * rng.normal()) / np.sqrt(2)])
    elif name == 'TDL-C':   # longer delay spread + Doppler
        pdp = np.exp(-np.arange(6) / 2.0)
        h = np.sqrt(pdp / pdp.sum()) * (rng.normal(size=6) + 1j * rng.normal(size=6)) / np.sqrt(2)
    y = np.convolve(x, h)[:N]
    if name == 'TDL-C':
        fd = rng.uniform(-1e-3, 1e-3)     # residual Doppler
        y = y * np.exp(1j * 2 * np.pi * fd * np.arange(N))
    return y

def awgn(y, snr_db, rng):
    p = np.mean(np.abs(y) ** 2)
    n0 = p / 10 ** (snr_db / 10)
    return y + np.sqrt(n0 / 2) * (rng.normal(size=N) + 1j * rng.normal(size=N))

def featurize(y):
    """Dual-view input: power-normalized raw IQ + log-magnitude spectrum + preamble-correlation IQ."""
    y = y / np.sqrt(np.mean(np.abs(y) ** 2, axis=-1, keepdims=True))
    r = y * np.conj(PRE)                      # de-rotate by known preamble
    spec = np.log(np.abs(np.fft.fft(y, axis=-1)) + 1e-3)
    spec = spec - spec.mean(axis=-1, keepdims=True)
    return np.concatenate([y.real, y.imag, r.real, r.imag, spec], axis=-1).astype(np.float32)

def generate(devs, dev_ids, chans, n_per, snr_range, seed):
    rng = np.random.default_rng(seed)
    X, Y, C, S, RAW = [], [], [], [], []
    for d in dev_ids:
        for ci, ch in enumerate(chans):
            for _ in range(n_per):
                snr = rng.uniform(*snr_range)
                y = awgn(channel(tx(devs[d], rng), ch, rng), snr, rng)
                RAW.append(y); Y.append(d); C.append(CHANNELS.index(ch)); S.append(snr)
    RAW = np.array(RAW)
    return featurize(RAW), np.array(Y), np.array(C), np.array(S), RAW

def augment_raw(raw, rng, snr_range=(0, 30)):
    """Channel augmentation applied at the receiver: random short FIR + Doppler + noise."""
    out = np.empty_like(raw)
    for i, y in enumerate(raw):
        L = rng.integers(1, 5)
        pdp = np.exp(-np.arange(L) / rng.uniform(0.5, 2.5))
        h = np.sqrt(pdp / pdp.sum()) * (rng.normal(size=L) + 1j * rng.normal(size=L)) / np.sqrt(2)
        h[0] = h[0] + rng.uniform(0, 2) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        z = np.convolve(y, h)[:N] * np.exp(1j * 2 * np.pi * rng.uniform(-1e-3, 1e-3) * np.arange(N))
        out[i] = awgn(z, rng.uniform(*snr_range), rng)
    return out

def perturb_impairments(raw, rng):
    """Impairment-Perturbation Outlier Synthesis (IPOS): turn received known bursts into
    virtual 'rogue transmitter' bursts by imposing an extra random hardware-impairment offset."""
    out = np.empty_like(raw); n = np.arange(N)
    for i, y in enumerate(raw):
        sgn = lambda: rng.choice([-1, 1])
        eps = sgn() * rng.uniform(0.03, 0.08); phi = sgn() * rng.uniform(2, 6) * np.pi / 180
        mu = np.cos(phi) + 1j * eps * np.sin(phi); nu = eps * np.cos(phi) - 1j * np.sin(phi)
        z = mu * y + nu * np.conj(y)
        a3 = rng.uniform(0.02, 0.08) * np.exp(1j * rng.uniform(-0.5, 0.5))
        z = z - a3 * z * np.abs(z) ** 2 / np.mean(np.abs(z) ** 2)
        z = z * np.exp(1j * 2 * np.pi * sgn() * rng.uniform(1e-3, 3e-3) * n)
        out[i] = z + 0.03 * (rng.normal() + 1j * rng.normal())
    return out

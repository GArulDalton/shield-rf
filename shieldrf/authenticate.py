"""Gateway-side authentication API (Algorithm 3 of the paper).

SHIELDAuthenticator wraps a trained SHIELD-RF model and provides
  * calibrate()   -> per-SNR-band energy thresholds tau_b (95% legitimate acceptance)
  * authenticate() -> device identity or REJECT for each burst
  * enroll()       -> retraining-free few-shot enrollment of a new device
"""
import numpy as np
from .model import infer

REJECT = -1
DEFAULT_BANDS = (0.0, 10.0, 20.0, 30.0 + 1e-9)   # SNR bands [0,10), [10,20), [20,30] dB


def _energy(logits):
    """Free energy E(v) = -log sum_k exp(l_k), Eq. (5)."""
    m = logits.max(1, keepdims=True)
    return -(np.log(np.exp(logits - m).sum(1)) + m[:, 0])


class SHIELDAuthenticator:
    def __init__(self, params, scale=16.0, bands=DEFAULT_BANDS, accept_rate=0.95):
        self.P = params
        self.scale = scale
        self.bands = np.asarray(bands)
        self.accept_rate = accept_rate
        protos = np.asarray(params['proto'])
        self.prototypes = protos / np.linalg.norm(protos, axis=1, keepdims=True)
        self.labels = list(range(len(self.prototypes)))   # device ids, extended by enroll()
        self.tau = None

    # -- helpers ---------------------------------------------------------
    def _band(self, snr_db):
        b = np.searchsorted(self.bands, np.asarray(snr_db), side='right') - 1
        return np.clip(b, 0, len(self.bands) - 2)

    def _embed_logits(self, X):
        z, _ = infer(self.P, X)
        return z, self.scale * z @ self.prototypes.T

    # -- Algorithm 3 -----------------------------------------------------
    def calibrate(self, X_val, snr_val):
        """tau_b = accept_rate quantile of legitimate energies in each SNR band."""
        _, lg = self._embed_logits(X_val)
        E, b = _energy(lg), self._band(snr_val)
        glob = np.quantile(E, self.accept_rate)
        self.tau = np.array([np.quantile(E[b == k], self.accept_rate) if np.any(b == k) else glob
                             for k in range(len(self.bands) - 1)])
        return self.tau

    def authenticate(self, X, snr_est):
        """Return (decisions, energies). decision = device id, or REJECT (-1)."""
        if self.tau is None:
            raise RuntimeError('call calibrate() first')
        _, lg = self._embed_logits(X)
        E = _energy(lg)
        ids = np.array(self.labels)[lg.argmax(1)]
        return np.where(E > self.tau[self._band(snr_est)], REJECT, ids), E

    def enroll(self, X_enroll, device_id):
        """Add a new device from K_s bursts: p_new = normalised mean embedding (no retraining)."""
        z, _ = infer(self.P, X_enroll)
        m = z.sum(0)
        self.prototypes = np.vstack([self.prototypes, m / np.linalg.norm(m)])
        self.labels.append(device_id)
        return len(self.labels) - 1

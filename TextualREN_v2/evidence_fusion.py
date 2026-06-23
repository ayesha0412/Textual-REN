"""
Evidence fusion: one presence posterior instead of a cascade of hard gates.

v3 rejected candidates at five independent hard gates (size filter, crop
score, confusable check, blur penalty, consensus) — each gate discards
information and errors compound (observed: a real pan rejected because
'tray' beat 'pan' by 0.004, well inside CLIP noise). v4 collects every
signal into an EvidenceVector and makes ONE decision:

    p = sigma(w . f + b)

    p >= tau_accept   -> confident FOUND
    p <  tau_abstain  -> NOT FOUND (calibrated abstention)
    in between        -> uncertainty zone (verifier-guided refinement, 1.6)

Weights live in configs/presence_weights.json. They start hand-initialized
and are fit ONCE on dev-split labels (Phase 4), then frozen — never tuned
per benchmark.
"""

import os
import json
import math
from dataclasses import dataclass, asdict


FEATURE_NAMES = [
    's_clip',    # CLIP crop-vs-query similarity of the chosen detection
    's_patch',   # patch-token spatial verification score
    'm_conf',    # effective confusable margin (target_eff - max confusable)
    'q_blur',    # frame blur quality in [0, 1] (Laplacian, normalized)
    'c_app',     # appearance consensus: mean pairwise CLIP-crop cosine
                 # across top verified candidates (replaces box-IoU, which
                 # measures camera motion, not agreement)
    's_gdino',   # Grounding DINO detection confidence
    's_ret',     # retrieval peak as z-score of the video's similarity dist
    'a_frac',    # bbox area fraction of frame (soft version of size filter)
    's_reid',    # exemplar re-ID similarity (0.0 until Contribution A lands)
    's_vlm',     # VLM verifier output in [-1, +1]: +1 = "YES this is the
                 # target", -1 = "NO it isn't", 0 = "UNSURE / skipped".
                 # Lazy-evaluated: only fired when m_conf < threshold (gate
                 # alignment, ~30-40% of queries). Non-redundant with CLIP
                 # because VLMs reason about object identity rather than
                 # contrastive similarity. Intervention E in the calibration
                 # plan; orthogonal to CERES (Liu et al. NeurIPS 2025).
]


@dataclass
class EvidenceVector:
    s_clip: float = 0.0
    s_patch: float = 0.0
    m_conf: float = 0.0
    q_blur: float = 0.0
    c_app: float = 0.0
    s_gdino: float = 0.0
    s_ret: float = 0.0
    a_frac: float = 0.0
    s_reid: float = 0.0
    s_vlm: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(float(v), 4) for k, v in asdict(self).items()}

    def to_list(self) -> list:
        return [float(getattr(self, name)) for name in FEATURE_NAMES]


class PresenceModel:
    """
    Logistic presence posterior p = sigma(w . f + b).

    Loads weights from a JSON file; falls back to hand-initialized defaults
    when the file is missing. fit() re-estimates weights from labeled
    evidence vectors (dev split only) and writes them back.
    """

    # Hand-initialized weights, scaled to each feature's natural range
    # (CLIP sims live in ~[0.1, 0.35], hence the large weight). These are
    # placeholders until dev labels exist — fit() replaces them.
    # s_ret weight is 0: the z-score of the distribution's MAX is always
    # large (max of ~7000 samples), and flatter distributions of ABSENT
    # objects produce the most extreme z (observed: absent "zebra" z=5.2 vs
    # ubiquitous "pan" z=1.3). The dev fit decides its true sign and scale.
    # s_vlm: +1.5 — VLM YES is full confidence, NO is damped to 0.3x
    # in vlm_verifier.py (crop wrong != object absent). Weight lowered
    # from 2.5 to 1.5 to prevent VLM from dominating fusion.
    _DEFAULT = {
        'weights': {
            's_clip': 8.0, 's_patch': 4.0, 'm_conf': 6.0, 'q_blur': 1.0,
            'c_app': 1.5, 's_gdino': 2.0, 's_ret': 0.0, 'a_frac': -1.0,
            's_reid': 1.5, 's_vlm': 1.5,
        },
        'bias': -3.5,
        'fitted': False,
        'note': 'hand-initialized defaults; fit on dev labels in Phase 4',
    }

    def __init__(self, weights_path: str = None):
        self.weights_path = weights_path
        data = None
        if weights_path and os.path.exists(weights_path):
            try:
                with open(weights_path, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = None
        if data is None:
            data = self._DEFAULT
        self.w = {k: float(data['weights'].get(k, 0.0)) for k in FEATURE_NAMES}
        self.b = float(data.get('bias', 0.0))
        self.fitted = bool(data.get('fitted', False))

    def presence(self, ev: EvidenceVector) -> float:
        """Presence posterior in (0, 1)."""
        z = self.b + sum(self.w[name] * float(getattr(ev, name))
                         for name in FEATURE_NAMES)
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, vectors: list, labels: list, save: bool = True,
             C: float = 1.0, class_weight=None) -> dict:
        """
        Fit logistic weights on labeled evidence vectors (DEV SPLIT ONLY).

        vectors:      list[EvidenceVector] or list[list[float]]
        labels:       list[int] (1 = correct/present, 0 = wrong/absent)
        C:            inverse regularization strength (lower = stronger).
                      Default 1.0 (sklearn default). Use C<1 when N is
                      small to prevent the fit collapsing to the prior.
        class_weight: passed through to sklearn. Use 'balanced' when the
                      positive:negative ratio exceeds ~3:1 so the loss
                      cares about the minority class instead of just
                      predicting the majority everywhere.

        Returns the fitted parameter dict; writes weights_path when save.
        """
        X = [v.to_list() if isinstance(v, EvidenceVector) else list(v)
             for v in vectors]
        y = [int(l) for l in labels]

        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, C=C,
                                       class_weight=class_weight)
            clf.fit(X, y)
            coefs = clf.coef_[0].tolist()
            bias = float(clf.intercept_[0])
        except ImportError:
            coefs, bias = self._fit_gd(X, y)

        self.w = dict(zip(FEATURE_NAMES, [float(c) for c in coefs]))
        self.b = bias
        self.fitted = True

        data = {
            'weights': self.w, 'bias': self.b, 'fitted': True,
            'n_samples': len(y),
            'note': 'fit on dev labels; FROZEN — do not refit per benchmark',
        }
        if save and self.weights_path:
            os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
            with open(self.weights_path, 'w') as f:
                json.dump(data, f, indent=2)
        return data

    @staticmethod
    def _fit_gd(X, y, lr=0.1, iters=2000):
        """Plain gradient-descent logistic fit (no-sklearn fallback)."""
        n_feat = len(X[0])
        w = [0.0] * n_feat
        b = 0.0
        n = len(X)
        for _ in range(iters):
            gw = [0.0] * n_feat
            gb = 0.0
            for xi, yi in zip(X, y):
                z = b + sum(wj * xj for wj, xj in zip(w, xi))
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                err = p - yi
                for j in range(n_feat):
                    gw[j] += err * xi[j]
                gb += err
            w = [wj - lr * gj / n for wj, gj in zip(w, gw)]
            b -= lr * gb / n
        return w, b

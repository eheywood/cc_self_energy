import numpy as np

def match_prep(x:np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    if np.iscomplexobj(x) and np.max(np.abs(x.imag)) < 1e-10:
        x = x.real
    return x[np.isfinite(x)]

def count_matches(spa, spin, label, atol=1e-4, rtol=0.0) -> tuple[int,int]:
    a = np.sort(match_prep(spa))
    b = np.sort(match_prep(spin))
    matches = np.any(np.isclose(a[:, None], b[None, :], atol=atol, rtol=rtol), axis=1)
    print(f"{label}: {int(matches.sum())}/{int(matches.size)} matched (atol={atol}, rtol={rtol})")
    return matches.sum(), matches.size

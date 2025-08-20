import numpy as np

molecules = [
    ("H 0.00 0.00 0.00; H 0.00 0.00 2.00", 'aug-cc-pVTZ', "H2"),
    ("Be 0.00000000 0.00000000 0.00000000", 'aug-cc-pVTZ', "Be"),
]

def match_prep(x:np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    if np.iscomplexobj(x) and np.max(np.abs(x.imag)) < 1e-10:
        x = x.real
    return x[np.isfinite(x)]

def count_matches(spa, spin, label, atol=1e-4, rtol=0.0) -> tuple[int,int, str]:
    a = np.sort(match_prep(spa))
    b = np.sort(match_prep(spin))
    matches = np.any(np.isclose(a[:, None], b[None, :], atol=atol, rtol=rtol), axis=1)

    text_results = f"{label}: {int(matches.sum())}/{int(matches.size)} matched (atol={atol}, rtol={rtol})"
    print(text_results)
    return matches.sum(), matches.size, text_results

def write_to_file(molecule: str, data: str, heading: bool = False) -> None:
    path = "./results/" + molecule + ".txt"
    with open (path, "a", encoding="utf-8") as f:
        if heading:
            stars = "*" * len(data)
            f.write(f"{stars}\n{data}\n{stars}\n")
        else:
            f.write(data + "\n")

  
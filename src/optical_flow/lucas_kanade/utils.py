from pathlib import Path


def getimgfiles(stem: Path, pat: str) -> list:
    stem = Path(stem).expanduser()
    flist = sorted([f for f in stem.glob(pat) if f.is_file()])
    if not flist:
        raise FileNotFoundError(f"no files found under {stem} using {pat}")
    return flist


def inside_circle(x, y, x_circ, y_circ, r):
    dx = abs(x - x_circ)
    dy = abs(y - y_circ)
    if dx > r:
        return False
    if dy > r:
        return False
    return True

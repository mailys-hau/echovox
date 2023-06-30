import numpy as np
import scipy.ndimage as scn

from warnings import warn



UNITS = {"meter": 1, "mm": 1e-3, "micron": 1e-6, "unknown": 1}

def resample_voxel_grid(nimg, new_res=[0.0005, 0.0005, 0.0005], dtype=np.uint8):
    # Expect `nib.Nift1Image
    old_res = np.array(nimg.header.get_zooms())[:3]
    conversion_rate = UNITS[nimg.header.get_xyzt_units()[0]]
    if conversion_rate == "unknown":
        warn("Unknown spatial unit, assume it's meter.", RuntimeWarning)
    old_res = 1e-3 * old_res
    grid = np.array(nimg.dataobj, dtype=dtype)
    ratio = old_res / new_res
    if grid.ndim == 4: # We're dealing with stacked onehot encodings
        ratio = np.insert(ratio, 0, 1)
    return scn.zoom(grid, ratio)

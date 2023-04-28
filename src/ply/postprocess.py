import numpy as np
import scipy.ndimage as sci




ndims = 3 # We work in 3D

# Connectivity of 1
STRUCT1 = sci.generate_binary_structure(ndims, 1)
# In between
STRUCT1_5 = np.stack([STRUCT1[1]] * 3)
# Full connectivity
STRUCT3 = sci.generate_binary_structure(ndims, 3)

POSTPROCESS = {"erosion": sci.binary_erosion, "dilation": sci.binary_dilation,
               "opening": sci.binary_opening, "closing": sci.binary_closing,
               "fill-holes": sci.binary_fill_holes}



def post_process(x, mode, structure=STRUCT1_5, iterations=10, mask=None, border_value=0, origin=0, brute_force=False):
    if mode is None:
        return x
    if mode == "fill-holes":
        return POSTPROCESS[mode](x, structure=structure, origin=origin)
    return POSTPROCESS[mode](x, structure=structure, iterations=iterations, mask=mask,
                             border_value=border_value, origin=origin, brute_force=brute_force)

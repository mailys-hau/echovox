import numpy as np

from dicoms.utils import frame2arr
from utils import LUT



def frames2vox(dcm_src, hdf, bbox, max_res, contrast):
    nb_frames = dcm_src.GetFrameCount()
    out = {}
    for f in range(nb_frames):
        frame = dcm_src.GetFrame(f, bbox, max_res)
        arr = frame2arr(frame)
        out[frame.time] = LUT[arr] if contrast else arr
    #FIXME? Assume same shape for every frame
    hdf["VolumeGeometry"].create_dataset("shape", data=list(out.values())[0].shape)
    # Don't save in HDF here in case you need to remove some frames
    return out

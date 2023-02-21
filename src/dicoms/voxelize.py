import numpy as np

from dicoms.utils import frame2arr



def frames2vox(dcm_src, hdf, bbox, max_res):
    nb_frames = dcm_src.GetFrameCount()
    frames, times = [], []
    saved = 1
    volumes = hdf["CartesianVolumes"]
    for f in range(nb_frames):
        frame = dcm_src.GetFrame(f, bbox, max_res)
        frames.append(frame2arr(frame)), times.append(frame.time)
    #FIXME? Assume same shape for every frame
    hdf["VolumeGeometry"].create_dataset("shape", data=frames[0].shape)
    # Don't save in HDF here in case you need to remove some frames
    return frames, np.array(times)
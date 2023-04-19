import numpy as np

from ply.voxelize import *




MODE = {"eigen": eigen_extrude, "normal": normal_extrude,
        "filter": filter_extrude, "region-growing": region_growing}



def plyseq2vox(sequence, frames, hdf, origin, directions, voxres, thickness, mode):
    """ Voxelize every frames' annotation in a sequence """
    # HDF file is expected to be open and close outside this function
    mesh2vox = MODE[mode]
    nbf = int(len(list(sequence.iterdir())) / 2) # Number of frames
    target = hdf.create_group("/GroundTruth")
    times = []
    for f, afname in zip(range(nbf), sequence.iterdir()):
        times.append(float(afname.stem.split('-')[1]))
        pfname = afname.with_stem(f"posterior-{times[-1]}")
        ant = mesh2vox(afname, frames[times[-1]], origin, directions, voxres, thickness)
        post = mesh2vox(pfname, frames[times[-1]], origin, directions, voxres, thickness)
        # Frames index start at 1
        target.create_dataset(f"anterior-{f + 1:02d}", data=ant)
        target.create_dataset(f"posterior-{f + 1:02d}", data=post)
    info = hdf["VolumeGeometry"]
    info.create_dataset("frameTimes", data=np.array(times))
    info.create_dataset("frameNumber", data=len(times))

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
    anteriors, posteriors = {}, {}
    for f, afname in zip(range(nbf), sequence.iterdir()):
        t = float(afname.stem.split('-')[1])
        times.append(t)
        pfname = afname.with_stem(f"posterior-{times[-1]}")
        anteriors[t] = mesh2vox(afname, frames[t], origin, directions, voxres, thickness)
        posteriors[t] = mesh2vox(pfname, frames[t], origin, directions, voxres, thickness)
    info = hdf["VolumeGeometry"]
    info.create_dataset("frameNumber", data=len(times))
    # There's no garanty iterdir sorts files, so we ensure it
    stimes = sorted(times)
    info.create_dataset("frameTimes", data=np.array(stimes))
    for i, t in enumerate(stimes):
        # Frames index start at 1
        target.create_dataset(f"anterior-{i + 1:02d}", data=anteriors[t])
        target.create_dataset(f"posterior-{i + 1:02d}", data=posteriors[t])

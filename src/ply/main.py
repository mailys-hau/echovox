import numpy as np

from ply.voxelize import eigen_extrude, normal_extrude



def plyseq2vox(sequence, hdf, origin, directions, voxres, voxshape, thickness, normals):
    """ Voxelize every frames in a sequence """
    # HDF file is expected to be open and close outside this function
    mesh2vox = normal_extrude if normals else eigen_extrude
    nbf = int(len(list(sequence.iterdir())) / 2) # Number of frames
    target = hdf.create_group("/GroundTruth")
    times = []
    for f, afname in zip(range(nbf), sequence.iterdir()):
        times.append(float(afname.stem.split('-')[1]))
        pfname = afname.with_stem(f"posterior-{times[-1]}")
        ant = mesh2vox(afname, voxres, voxshape, origin, directions, thickness)
        post = mesh2vox(pfname, voxres, voxshape, origin, directions, thickness)
        # Frames index start at 1
        target.create_dataset(f"anterior-{f + 1:02d}", data=ant)
        target.create_dataset(f"posterior-{f + 1:02d}", data=post)
    info = hdf["VolumeGeometry"]
    info.create_dataset("frameTimes", data=np.array(times))
    info.create_dataset("frameNumber", data=len(times))

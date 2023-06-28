"""
Convert HDFs to NIFTIs and vice versa
"""

import click as cli
import h5py
import nibabel as nib
import numpy as np

from pathlib import Path

from utils import get_fname



def _hdf2nii(fname, idir, gtdir):
    hdf = h5py.File(fname, 'r')
    directions = hdf["VolumeGeometry"]["directions"][()]
    origin = np.expand_dims(hdf["VolumeGeometry"]["origin"][()], 0).T
    affine = np.vstack([np.hstack([directions, origin]), np.array([0, 0, 0, 1])])
    for i in range(hdf["VolumeGeometry"]["frameNumber"][()]):
        iname = get_fname(fname, idir, ".nii", i)
        gtname = get_fname(fname, gtdir, ".nii", i)
        # TODO? Add more info in header
        iimg = nib.Nifti1Image(hdf["CartesianVolume"][f"vol{i + 1:02d}"][()], affine)
        ant = hdf["GroundTruth"][f"anterior-{i + 1:02d}"][()].astype(np.uint8)
        post = hdf["GroundTruth"][f"posterior-{i + 1:02d}"][()].astype(np.uint8)
        gt = np.stack([ant, post])
        gtimg = nib.Nifti1Image(gt, affine)
        nib.save(iimg, iname)
        nib.save(gtimg, gtname)
    hdf.close()


@cli.group(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
def main():
    pass


@main.command(name="nii2hdf", short_help="ToDo.")
def nii2hdf():
    pass

@main.command(name="hdf2nii", short_help="Convert HDFs to NIFTIs:")
@cli.argument("hdfdir", type=cli.Path(exists=True, resolve_path=True, path_type=Path, file_okay=False))
@cli.option("--input-directory", "-i", "idir", default="inputs",
            type=cli.Path(resolve_path=True, path_type=Path, file_okay=False),
            help="Where to store input voxel grids.")
@cli.option("--ground-truth-directory", "-gt", "gtdir", default="ground-truth",
            type=cli.Path(resolve_path=True, path_type=Path, file_okay=False),
            help="Where to store ground truth segmentation mask.")
def hdf2nii(hdfdir, idir, gtdir):
    """
    Convert HDFs containing multiple volumes to several NIFTIs each containing one
    volume. Inputs and ground truth are stored in separate directories.

    HDFDIR    PATH    Directory of HDFs containing voxels.
    """
    idir.mkdir(parents=True, exist_ok=True), gtdir.mkdir(parents=True, exist_ok=True)
    # TODO: Multiprocess this loop
    for fname in hdfdir.iterdir():
        if fname.suffix != ".h5":
            print(f"Skipping {fname.name}, not an HDF.")
            continue
        _hdf2nii(fname, idir, gtdir)



if __name__ == "__main__":
    main()

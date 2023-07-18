"""
Convert HDFs to NIfTIs and vice versa. Tool box for NTNU-Polimi collaboration.
"""

import click as cli
import h5py
import nibabel as nib
import numpy as np

from pathlib import Path
from tqdm.contrib.concurrent import thread_map

from utils import get_affine, get_fname, resample_voxel_grid, to_onehot, to_labels



def _nii2hdf(iname, gtdir, hdfdir, scaling=[0.0005, 0.0005, 0.0005]):
    if iname.suffix != ".nii":
        print(f"Skipping {iname.name}, not an NIfTI.")
        return
    # idir in contained in iname
    hname = get_fname(iname, hdfdir, ".h5")
    gtname = get_fname(iname, gtdir, ".nii")
    frame = "01" # There's only one frame in the given NIfTIs
    scaling = np.array(scaling)
    # Save input
    iimg = nib.load(iname)
    iarr = resample_voxel_grid(iimg, scaling)
    hdf = h5py.File(hname, 'w')
    inp = hdf.create_group("CartesianVolume")
    inp.create_dataset(f"vol{frame}", data=iarr)
    # Save ground truth
    gt = hdf.create_group("GroundTruth")
    gtimg = nib.load(gtname)
    gtarr = np.array(gtimg.dataobj, dtype=np.uint8)
    # Receive label encoding with 1=mitral annulus, 2=anterior, 3=posterior
    onehot = nib.Nifti1Image(to_onehot(gtarr, [0, 1]), gtimg.affine, gtimg.header)
    onehot = resample_voxel_grid(onehot, scaling, bool, order=0)
    gt.create_dataset(f"anterior-{frame}", data=onehot[0])
    gt.create_dataset(f"posterior-{frame}", data=onehot[1])
    # Save additional information
    info = hdf.create_group("VolumeGeometry")
    info.create_dataset("frameNumber", data=int(frame))
    info.create_dataset("directions", data=iimg.affine[:3, :3])
    info.create_dataset("origin", data=iimg.affine[:3, -1])
    info.create_dataset("resolution", data=scaling)
    hdf.close()

def _hdf2nii(fname, idir, gtdir, middle):
    if fname.suffix != ".h5":
        print(f"Skipping {fname.name}, not an HDF.")
        return
    hdf = h5py.File(fname, 'r')
    directions = hdf["VolumeGeometry"]["directions"][()]
    spacing = hdf["VolumeGeometry"]["resolution"][()]
    affine = get_affine(directions, spacing)
    header = nib.Nifti1Header()
    header.set_xyzt_units(xyz=1) # Set unit to meter
    if middle: # Only convert middle frame
        idx = int(hdf["VolumeGeometry"]["frameNumber"][()] / 2) + 1
        iname, gtname = get_fname(fname, idir, ".nii"), get_fname(fname, gtdir, ".nii")
        # TODO? Add more info in header (directions, origin)
        iimg = nib.Nifti1Image(hdf["CartesianVolume"][f"vol{idx:02d}"][()], affine, header=header)
        ant = hdf["GroundTruth"][f"anterior-{idx:02d}"][()].astype(np.uint8)
        post = hdf["GroundTruth"][f"posterior-{idx:02d}"][()].astype(np.uint8)
        gt = np.stack([ant, post])
        gtimg = nib.Nifti1Image(to_labels(gt), affine, header=header)
        nib.save(iimg, iname)
        nib.save(gtimg, gtname)
    else:
        for i in range(hdf["VolumeGeometry"]["frameNumber"][()]):
            iname = get_fname(fname, idir, ".nii", i)
            gtname = get_fname(fname, gtdir, ".nii", i)
            # TODO? Add more info in header
            iimg = nib.Nifti1Image(hdf["CartesianVolume"][f"vol{i + 1:02d}"][()], affine)
            ant = hdf["GroundTruth"][f"anterior-{i + 1:02d}"][()].astype(np.uint8)
            post = hdf["GroundTruth"][f"posterior-{i + 1:02d}"][()].astype(np.uint8)
            gt = np.stack([ant, post])
            gtimg = nib.Nifti1Image(to_labels(gt), affine)
            nib.save(iimg, iname)
            nib.save(gtimg, gtname)
    hdf.close()


@cli.group(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
def main():
    pass


@main.command(name="nii2hdf", short_help="Convert NIfTIs to HDFs.")
@cli.argument("idir", type=cli.Path(exists=True, resolve_path=True, path_type=Path, file_okay=False))
@cli.argument("gtdir", type=cli.Path(exists=True, resolve_path=True, path_type=Path, file_okay=False))
@cli.option("--hdf-directory", "-d", "hdfdir", default="hdf",
            type=cli.Path(resolve_path=True, path_type=Path, file_okay=False),
            help="Where to store all voxel grids.")
@cli.option("--scaling", "-s", type=cli.Tuple([cli.FloatRange(min=0)] * 3),
            nargs=3, default=[0.0005] * 3, help="Resolution of a voxel in meter.")
@cli.option("--number-workers", "-n", "nb_workers", type=cli.IntRange(min=1), default=1,
            help="Number of workers used to accelerate file processing.")
def nii2hdf(idir, gtdir, hdfdir, scaling, nb_workers):
    hdfdir.mkdir(parents=True, exist_ok=True)
    nbfiles = len(list(idir.glob("*.nii")))
    thread_map(lambda fname: _nii2hdf(fname, gtdir, hdfdir, scaling), idir.iterdir(), max_workers=nb_workers,
               # Pretty progress bar
               desc="Processed", unit="files", total=nbfiles, colour="green")

@main.command(name="hdf2nii", short_help="Convert HDFs to NIfTIs.")
@cli.argument("hdfdir", type=cli.Path(exists=True, resolve_path=True, path_type=Path, file_okay=False))
@cli.option("--image-directory", "-i", "idir", default="images",
            type=cli.Path(resolve_path=True, path_type=Path, file_okay=False),
            help="Where to store input voxel grids.")
@cli.option("--mask-directory", "-m", "gtdir", default="masks",
            type=cli.Path(resolve_path=True, path_type=Path, file_okay=False),
            help="Where to store ground truth segmentation mask.")
@cli.option("--only-middle-frame", "-o/ ", "middle", is_flag=True, default=False,
            help="Only convert middle frame contained in an HDF.")
@cli.option("--number-workers", "-n", "nb_workers", type=cli.IntRange(min=1), default=1,
            help="Number of workers used to accelerate file processing.")
def hdf2nii(hdfdir, idir, gtdir, middle, nb_workers):
    """
    Convert HDFs containing multiple volumes to several NIfTIs each containing one
    volume. Inputs and ground truth are stored in separate directories.

    HDFDIR    PATH    Directory of HDFs containing voxels.
    """
    idir.mkdir(parents=True, exist_ok=True), gtdir.mkdir(parents=True, exist_ok=True)
    nbfiles = len(list(hdfdir.glob("*.h5")))
    thread_map(lambda fname: _hdf2nii(fname, idir, gtdir, middle), hdfdir.iterdir(), max_workers=nb_workers,
               # Pretty progress bar
               desc="Processed", unit="files", total=nbfiles, colour="green")



if __name__ == "__main__":
    main()

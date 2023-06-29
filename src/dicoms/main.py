"""
This script needs to be run in a Windows shell and assumes Image3dAPI is in place and that
GEHC_CARD_US reader's DDL is registered.
"""
import click as cli
import h5py
import numpy as np

from pathlib import WindowsPath
from pythoncom import CoInitialize
from tqdm.contrib.concurrent import thread_map

from dicoms.loaders import load_dcm, load_dcm_info
from dicoms.utils import save_selected_frames
from dicoms.voxelize import frames2vox



def dcmseq2vox(dcm_src, hdf, voxres, bbox, contrast):
    info = hdf["VolumeGeometry"]
    if "frameTimes" in info.keys():
        # Voxelize only listed frames (aka annotated close valve)
        cond = lambda ftime: ftime in info["frameTimes"][()]
    else:
        cond = lambda ftime: True # Voxelize every frames
    res = np.round(np.linalg.norm(info["directions"], axis=1) / voxres)
    max_res = np.ctypeslib.as_ctypes(res.astype(np.ushort))
    info.create_dataset("resolution", data=voxres)
    return frames2vox(dcm_src, hdf, bbox, max_res, contrast)



def _multiprocess(dname, opath, voxres):
    """ Wrapper around `dcmseq2vox` """
    CoInitialize() # Needed to work with comtypes and multithread
    if dname.suffix != ".dcm":
        print(f"Ignoring {dname.name}, not a DICOM.")
        return
    hname = opath.joinpath(dname.with_suffix(".h5").name)
    hdf = h5py.File(hname, 'a') # In case annotations were done first
    dcm_src = load_dcm(dname)
    bbox = load_dcm_info(dcm_src, hdf) # Store ECG, origin, directions, ...
    frames, times = dcmseq2vox(dcm_src, hdf, voxres, bbox)
    save_selected_frames(frames, hdf, times)
    hdf.close()



@cli.command(context_settings={"help_option_names": ["--help", "-h"], "show_default": True})
@cli.argument("dicomdir", type=cli.Path(exists=True, resolve_path=True, path_type=WindowsPath))
@cli.option("--voxel-resolution", "-r", "voxres", type=cli.Tuple([cli.FloatRange(min=0)] * 3),
            nargs=3, default=[0.0007] * 3, help="Resolution of a voxel in meter.")
@cli.option("--output-directory", "-o", "opath",
            type=cli.Path(resolve_path=True, path_type=WindowsPath), default="voxels",
            help="Where to store generated voxel grids.")
@cli.option("--number-workers", "-n", "nb_workers", default=1, type=cli.IntRange(min=1),
            help="Number of workers used to accelerate file processing.")
def dcm2vox(dicomdir, voxres, opath, nb_workers):
    """
    Convert GE DICOMs to HDF. Save each frames with the given resolution. Voxel grid shape will
    depend of the data since the resolution is fixed.

    DICOMDIR    PATH    Directory of DICOMs to convert to HDFs.
    """
    opath.mkdir(exist_ok=True)
    voxres, voxshape = np.array(voxres), np.array(voxshape)
    nb_files = len(list(dcmdir.glob("*.dcm")))
    # Allow multithread with nice progress bar
    thread_map(lambda fname: _multiprocess(fname, opath, voxres),
               dcmdir.iterdir(), max_workers=nb_workers,
               # Pretty loading bar
               desc="Processed", unit="DICOM", total=nb_files, colour="green")



if __name__ == "__main__":
    dcm2vox()

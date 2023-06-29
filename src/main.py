import click as cli
import h5py
import numpy as np

from pathlib import WindowsPath
from pythoncom import CoInitialize
from tqdm.contrib.concurrent import thread_map

from dicoms import dcmseq2vox
from dicoms.loaders import load_dcm, load_dcm_info
from dicoms.utils import save_selected_frames
from ply import plyseq2vox



def seq2vox(dname, pdir, opath, voxres, thickness, mode, contrast, postprocess):
    CoInitialize() # Needed to work with comtypes and multithread
    if dname.suffix != ".dcm":
        print(f"Ignoring {dname.name}, not a DICOM.")
        return
    hdf = h5py.File(opath.joinpath(dname.with_suffix(".h5").name), 'w')
    src = load_dcm(dname)
    bbox = load_dcm_info(src, hdf)
    info = hdf["VolumeGeometry"]
    # Voxelize inputs
    frames = dcmseq2vox(src, hdf, voxres, bbox, contrast)
    # Will voxelize and add to HDF, ground truth, frame times and number of frame
    plyseq2vox(pdir.joinpath(dname.stem), frames, hdf, info["origin"][()],
               info["directions"][()], voxres, thickness, mode, postprocess)
    # Save only frame that have an annotation
    save_selected_frames(frames, hdf)
    hdf.close()


@cli.command(context_settings={"help_option_names": ["--help", "-h"], "show_default": True})
@cli.argument("plydir", type=cli.Path(exists=True, resolve_path=True, path_type=WindowsPath,
              file_okay=False))
@cli.argument("dcmdir", type=cli.Path(exists=True, resolve_path=True, path_type=WindowsPath,
              file_okay=False))
@cli.option("--voxel-resolution", "-r", "voxres", type=cli.Tuple([cli.FloatRange(min=0)] * 3),
            nargs=3, default=[0.0007] * 3, help="Resolution of a voxel in meter.")
@cli.option("--thickness", "-t", type=cli.FloatRange(min=0), default=0.003,
            help="Thickness of extruded leaflets' segmentation in meter.")
@cli.option("--extrusion-mode", "-m", "mode", default="normal",
            type=cli.Choice(["eigen", "normal", "filter", "region-growing"], case_sensitive=False),
            help="Which extrusion method to use (see README.txt).")
@cli.option("--contrast/--no-contrast", "-c/ ", is_flag=True, default=False,
            help=("Whether to use lookup table to enhance input contrast."
                  " If one is contained in the DICOM, use it, otherwise, use a generic one."))
@cli.option("--postprocess", "-p",
            type=cli.Choice(["erosion", "dilation", "opening", "closing", "fill-holes"], case_sensitive=False),
            help=("If you want some binary post-processing on the annotation voxel grid. "
                  "This is useful for filter and region-growing extrusion."))
@cli.option("--ouput-directory", "-o", "opath", type=cli.Path(resolve_path=True,
            path_type=WindowsPath, file_okay=False), default="voxels",
            help="Where to store generated voxels.")
@cli.option("--number-workers", "-n", "nb_workers", type=cli.IntRange(min=1), default=1,
            help="Number of workers used to accelerate file processing.")
def all2vox(plydir, dcmdir, voxres, thickness, mode, contrast, postprocess, opath, nb_workers):
    """
    Convert given DICOMs and associated triangle meshes to voxel grids. Inputs are expected to
    be grouped by sequence. Results will be stored in `output-directory/sequence-name.h5`.
    Meshes are extruded of `thickness` and voxelized in a grid of given resolution *and* shape.
    DICOMS are voxelized following the given resolution (shape will be arbitrary).

    \b
    PLYDIR    PATH    Directory of triangle meshes.
    DCMDIR    PATH    Directory of input dicoms (3D TEE).
    """
    opath.mkdir(exist_ok=True)
    voxres = np.array(voxres)
    nb_sequences = len(list(dcmdir.glob("*.dcm")))
    thread_map(lambda fname: seq2vox(fname, plydir, opath, voxres, thickness, mode,
                                     contrast, postprocess),
               dcmdir.iterdir(), max_workers=nb_workers,
               # Pretty loading bar
               desc="Processed", unit="sequence", total=nb_sequences, colour="green")



if __name__ == "__main__":
    all2vox()

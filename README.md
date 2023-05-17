# EchoVox
## From 3D ultrasound and surface mesh to voxel grid.
Convert 3D TEE to voxel grid and extrude mitral annulus surface to voxel grid automatically.

## ToDos
- [ ] Add `requirements.txt`
- [x] Add description of extrusion methods


## Requirements
See `requirements.txt` for all needed Python packages. The scripts are made to run on a Windows system and require Python 3.9.

To properly extract the DICOMs file you will need the vendor's Image3dAPI (see [here](https://github.com/MedicalUltrasound/Image3dAPI) for more information).

We only tested this scripts with General Eletric API.


## Usage
#### Convert DICOM and PLY
`$ python main.py <path/to/ply/files/> <path/to/dicom/files/> [OPTIONS]`. For more information see `$ python main.py -h`.

This will convert all the frames in an achocardiogram **that are linked to a surface mesh** to voxel grids and save them in HDF files, along with necessary information. The given directories should abide by the following structure:
```
dicoms/
|-- foo.dcm
|-- bar.dcm
|-- ...
ply/
|-- foo/
    |-- anterior-franetime1.ply
    |-- anterior-franetime2.ply
    |-- ...
    |-- posterior-franetime1.ply
    |-- ...
|-- bar/
    |-- ...
|-- ...
```
#### Convert DICOM
`$ python main.py <path/to/dicom/files/> [OPTIONS]`. For more information see `$ python dicoms/main.py -h`.

This will convert all frames in an echocardiogram to voxel grids and save them in HDF files, along with necessary information.

#### Convert PLY
This option is not available as not all frame from a DICOM are annotated.


## Extrusion
Several methods are available to extrude a surface mesh to a volume. Each method assume that the surface is located in the middle of the leaflet and extrude of half the given thickness in each directions of the extrusion vector. It is possible (and recommended) to use some [morphological operation](https://en.wikipedia.org/wiki/Mathematical_morphology) to amend the potential holes in the produced volume (you can specify this option to the main script).
#### From eigen vector
The extrusion vector is eigen vector of the mesh. This assumption works since the mesh is the valve surface, and we want to get the volume of the leaflets. The same vector is used along the full surface.
#### From normal vectors
The extrusion vectors are the normal vectors of the mesh. Leads to a more suitable volume than the eigen method.
#### Imitating the region growind algorithm
The volume is first obtained using the normal method, and then refined using the [region growing algorithm](https://en.wikipedia.org/wiki/Region_growing). The seed is the intensity of the echocardiogram voxels located on the surface mesh, and we keep all voxel in the volume if their itensity is above `mean - std` of the seed.

This method gives the best looking volume.


## Output
Each DICOMs is converted to an isotropic voxel grid (0.7 mm by default). Each PLY surface mesh is extruded and converted to a voxel grid of same size and resolution as its paired DICOM.

All extracted information are saved in HDF files, one per sequence. Only frames linked to an annotation are saved in the final HDF files.
HDF's structure:
```
|-- CartesianVolume/
    |-- vol01
    |-- vol02
    |-- ...
|-- ECG/
    |-- samples
    |-- times
|-- GroundTruth/
    |-- anterior-01
    |-- anterior-02
    |-- ...
    |-- posterior-01
    |-- ...
|-- VolumeGeometry/
    |-- directions
    |-- frameNumber
    |-- frameTimes
    |-- origin
    |-- resolution
    |-- shape
|-- colorMap
```

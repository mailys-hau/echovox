"""
Code from Gabriel Kiss 01.2020
"""

import comtypes.client as ccomtypes
import numpy as np
import platform

from pathlib import WindowsPath

from dicoms.utils import safe2np





### Change this according to your system ###
Image3DAPIWin32 = None
Image3DAPIx64 = WindowsPath("C:/Users/malou/Documents/dev/Image3dAPI/x64/Image3dAPI.tlb")



def load_dcm(fname):
    # Load type library
    if "32" in platform.architecture()[0]:
        Image3dAPI = ccomtypes.GetModule(str(Image3DAPIWin32))
    else:
        Image3dAPI = ccomtypes.GetModule(str(Image3DAPIx64))
    # Create loader object
    loader = ccomtypes.CreateObject("GEHC_CARD_US.Image3dFileLoader")
    loader = loader.QueryInterface(Image3dAPI.IImage3dFileLoader)
    # Load file
    err_type, err_msg = loader.LoadFile(str(fname)) #TODO? Print errors
    return loader.GetImageSource()

def load_dcm_info(src, hdf):
    #probe = src.GetProbeInfo() #TODO? Should be saved
    # Retrive ECG info
    ecg = src.GetECG()
    samples = safe2np(ecg.samples)
    trig_time = safe2np(ecg.trig_times)
    group_ecg = hdf.create_group("/ECG")
    group_ecg.create_dataset("samples", data=samples)
    group_ecg.create_dataset("times", data=np.linspace(trig_time[0], trig_time[1], 
                                                       num=samples.shape[0]))
    # Get bounding box (GE uses dir2 as deepness axis)
    bbox = src.GetBoundingBox()
    origin = np.array([bbox.origin_x, bbox.origin_y, bbox.origin_z])
    dir_x = np.array([bbox.dir1_x, bbox.dir1_y, bbox.dir1_z])
    dir_y = np.array([bbox.dir2_x, bbox.dir2_y, bbox.dir2_z])
    dir_z = np.array([bbox.dir3_x, bbox.dir3_y, bbox.dir3_z])
    info = hdf.create_group("/VolumeGeometry")
    info.create_dataset("origin", data=origin)
    info.create_dataset("directions", data=np.stack([dir_x, dir_y, dir_z]))
    # Save color map
    cmap = src.GetColorMap()
    hdf.create_dataset("colorMap", data=src.GetColorMap())
    return bbox

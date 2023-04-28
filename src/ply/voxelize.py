import numpy as np
import trimesh as tm

from ply.utils import full_load_ply



def get_vox_idx(mesh, inc, voxshape, origin, directions, voxres):
    is_inside = lambda i: np.all((0 <= i) & (i < voxshape), axis=1)
    # Subdivide mesh so you have at least one point per voxel
    verts, _  = tm.remesh.subdivide_to_size(mesh.vertices + inc, mesh.faces,
                                            max_edge=voxres / 2, max_iter=20)
    # Translate to DICOMs (input) coordinate. Directions given by line => right multiplication
    verts = (verts - origin) @ np.linalg.inv(directions)
    # Convert to voxel indexes (that represent leaflets)
    idx = np.round(verts * voxshape).astype(int) # Scale to number of voxel
    # Ensure all indexes are inside input voxel grid
    return np.unique(idx[is_inside(idx)], axis=0)

def _extrude(mesh, voxshape, origin, directions, voxres, extrude_vec, extrude=0.003):
    """ Code from Sverre Herland """
    # Voxelize every increment around surface with proper spacing to get a full volume
    increments = np.stack([np.arange(-extrude / 2, extrude / 2, vr) for vr in voxres])
    allidx = [] # Accumulate points in subdivided meshes, it'll be your positive voxels
    for i in range(increments.shape[-1]):
        inc = increments[:,i]
        allidx.append(get_vox_idx(mesh, inc * extrude_vec, voxshape, origin, directions, voxres))
    allidx = np.unique(np.concatenate(allidx, axis=0), axis=0)
    voxel_grid = np.zeros(voxshape, dtype=bool)
    # Set voxels inside leaflets to True
    voxel_grid[allidx[:, 0], allidx[:, 1], allidx[:, 2]] = 1
    return voxel_grid


def eigen_extrude(fname, vinput, origin, directions, voxres, extrude=0.003):
    """
    Code from Sverre Herland
    Extrude along the smallest eighen vector of half `extrude` value in each direction.
    """
    with open(fname, "br") as fd: # Need to be opened in binary mode for Trimesh
        dict_mesh = tm.exchange.ply.load_ply(fd, prefer_color="face")
    mesh = tm.Trimesh(**dict_mesh)
    covariance = np.cov(mesh.vertices.T)
    eig_vals, eig_vecs = np.linalg.eig(covariance)
    extrude_vec = eig_vecs[np.argmin(eig_vals)] # Should be Y-axis
    return _extrude(mesh, vinput.shape, origin, directions, voxres, extrude_vec, extrude)

def normal_extrude(fname, vinput, origin, directions, voxres, extrude=0.003):
    """
    Extrude along each vertices normals of half `extrude` value in each direction of the normal.
    This method yields a more accurate volume than the `eighen_extrude`
    """
    with open(fname, "br") as fd: # Need to be opened in binary mode for Trimesh
        dict_mesh = full_load_ply(fd, prefer_color="face")
    mesh = tm.Trimesh(**dict_mesh)
    return _extrude(mesh, vinput.shape, origin, directions, voxres, mesh.vertex_normals, extrude)

def filter_extrude(fname, vinput, origin, directions, voxres, extrude=0.003, div=1):
    """
    Extrude along each vertices normals of half `extrude` value in each direction of the normal.
    Then refine volume by filtering out outlier voxels with intensity out of mean ± std.
    """
    with open(fname, "br") as fd:
        dict_mesh = full_load_ply(fd, prefer_color="face")
    mesh = tm.Trimesh(**dict_mesh)
    voxshape = vinput.shape
    # Bounding box annotation
    box = _extrude(mesh, voxshape, origin, directions, voxres, mesh.vertex_normals, extrude)
    idx = np.argwhere(box) # Box is a boolean array
    out = np.zeros_like(box)
    rmin, rmax = np.min(idx, axis=0), np.max(idx, axis=0) # Indexes range of surface
    strides = ((rmax - rmin) / div).astype(int)
    keep = lambda x: (mean - std <= x)# & (x <= mean + std)
    is_inside = lambda l: np.all((inf <= l) & (l < sup), axis=1)
    bound = lambda x, axis: min(x + strides[axis], voxshape[axis])
    # Filter voxels
    for i in range(rmin[0], rmax[0], max(strides[0], 1)):
        for j in range(rmin[1], rmax[1], max(strides[1], 1)):
            for k in range(rmin[2], rmax[2], max(strides[2], 1)):
                # Subdivided seed for more precision
                inf, sup = (i, j, k), (bound(i, 0), bound(j, 1), bound(k, 2))
                sidx = np.unique(idx[is_inside(idx)], axis=0)
                if sidx.size == 0:
                    continue
                seed = vinput[sidx[:,0], sidx[:,1], sidx[:,2]]
                mean, std = np.mean(seed), np.std(seed)
                # Filter annotation to contain voxels close in brightness to surface only
                out[sidx] = np.where(keep(vinput[sidx]), box[sidx], False)
    return out

def region_growing(fname, vinput, origin, directions, voxres, extrude=0.003, div=2):
    """
    Extrude along each vertices normals of half `extrude` value in each direction of the normal.
    Then refine volume by keeping voxels which intensity is close enough to the *surface* ones (mean ± std).
    """
    with open(fname, "br") as fd:
        dict_mesh = full_load_ply(fd, prefer_color="face")
    mesh = tm.Trimesh(**dict_mesh)
    voxshape = vinput.shape
    # Bounding box annotation
    box = _extrude(mesh, voxshape, origin, directions, voxres, mesh.vertex_normals, extrude)
    idx = get_vox_idx(mesh, 0, voxshape, origin, directions, voxres) # Surface voxels
    out = np.zeros_like(box)
    rmin, rmax = np.min(idx, axis=0), np.max(idx, axis=0) # Indexes range of surface
    strides = ((rmax - rmin) / div).astype(int)
    keep = lambda x: (mean - std <= x)# & (x <= mean + std)
    is_inside = lambda l: np.all((inf <= l) & (l < sup), axis=1)
    bound = lambda x, axis: min(x + strides[axis], voxshape[axis])
    # Filter voxels
    for i in range(rmin[0], rmax[0], max(strides[0], 1)):
        for j in range(rmin[1], rmax[1], max(strides[1], 1)):
            for k in range(rmin[2], rmax[2], max(strides[2], 1)):
                # Subdivided seed for more precision
                inf, sup = (i, j, k), (bound(i, 0), bound(j, 1), bound(k, 2))
                sidx = np.unique(idx[is_inside(idx)], axis=0)
                if sidx.size == 0:
                    continue
                seed = vinput[sidx[:,0], sidx[:,1], sidx[:,2]]
                mean, std = np.mean(seed), np.std(seed)
                # Filter annotation to contain voxels close in brightness to surface only
                out[sidx] = np.where(keep(vinput[sidx]), box[sidx], False)
    return out

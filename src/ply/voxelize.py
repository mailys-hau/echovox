import numpy as np
import trimesh as tm

from ply.utils import full_load_ply



def _extrude(mesh, voxres, voxshape, origin, directions, extrude_vec, extrude=0.003):
    """ Code from Sverre Herland """
    # Voxelize every increment around surface with proper spacing to get a full volume
    increments = np.stack([np.arange(-extrude / 2, extrude / 2, vr) for vr in voxres])
    allidx = [] # Accumulate points in subdivided meshes, it'll be your positive voxels
    is_inside = lambda i: np.all((0 <= i) & (i < voxshape), axis=1)
    for i in range(increments.shape[-1]):
        inc = increments[:,i]
        # Subdivide mesh so you have at least a point per voxel
        verts, _ = tm.remesh.subdivide_to_size(mesh.vertices + inc * extrude_vec, mesh.faces,
                                               max_edge=voxres / 2, max_iter=20)
        # Translate to DICOMs coordinate, directions given by line => right multiplication
        verts = (verts - origin) @ np.linalg.inv(directions)
        # Convert to voxel indexes (the one that represent leaflets)
        idx = np.round(verts * voxshape).astype(int) # Scale to number of voxel
        allidx.append(np.unique(idx[is_inside(idx)], axis=0))
    allidx = np.unique(np.concatenate(allidx, axis=0), axis=0)
    voxel_grid = np.zeros(voxshape, dtype=bool)
    # Set voxels inside leaflets to True
    voxel_grid[allidx[:, 0], allidx[:, 1], allidx[:, 2]] = 1
    return voxel_grid


def eigen_extrude(fname, voxres, voxshape, origin, directions, extrude=0.003):
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
    return _extrude(mesh, voxres, voxshape, origin, directions, extrude_vec, extrude)

def normal_extrude(fname, voxres, voxshape, origin, directions, extrude=0.003):
    """
    Extrude along each vertices normals of half `extrude` value in each direction of the normal.
    This method yields a more accurate volume than the `eighen_extrude`
    """
    with open(fname, "br") as fd: # Need to be opened in binary mode for Trimesh
        dict_mesh = full_load_ply(fd, prefer_color="face")
    mesh = tm.Trimesh(**dict_mesh)
    return _extrude(mesh, voxres, voxshape, origin, directions, mesh.vertex_normals, extrude)


def _rg_extrude(fname, voxres, voxshape, origin, directions, extrude=0.003):
    with open(fname, "br") as fd:
        dict_mesh = full_load_ply(fd, prefer_color="face")
    mesh = tm.Trimesh(**dict_mesh)
    # Bounding box annotation
    box = _extrude(mesh, voxres, voxshape, origin, directions, mesh.vertex_normals, extrude)
    # Get voxels containing the *surface*, and extract their voxel intensity
    is_inside = lambda i: np.all((0 <= i) & (i < voxshape), axis=1)
    verts, _ = tm.remesh.subdivide_to_size(mesh.vertices, mesh.faces,
                                           max_edge=voxres / 2, max_iter=20)
    verts = (verts - origin) @ np.linalg.inv(directions)
    idx = np.round(verts * voxshape).astype(int)
    idx = np.unique(idx[is_inside(idx)], axis=0) # Ensure it's inside the voxel grid
    return box, idx

def region_growing_extrude(fname, voxres, voxshape, origin, directions, vinput, extrude=0.003):
    """
    Extrude along each vertices normals of half `extrude` value in each direction of the normal.
    Then refine volume by keeping voxels which intensity is close enough to the surface ones (mean ± std).
    """
    box, idx = _rg_extrude(fname, voxres, voxshape, origin, directions, extrude=extrude)
    seed = vinput[idx[:,0], idx[:,1], idx[:,2]]
    mean, std = np.mean(seed), np.std(seed)
    # Filter annotation to contain voxels close in brightness to surface only
    cond = lambda x: (mean - std <= x) & (x <= mean + std)
    return np.where(cond(vinput), box, False)

def subdiv_region_growing_extrude(fname, voxres, voxshape, origin, directions, vinput,
                                  extrude=0.003, div=5):
    """
    Extrude along each vertices normals of half `extrude` value in each direction of the normal.
    Then refine volume by keeping voxels which intensity is close enough to the surface ones (mean ± std).
    Here, mean and std are computed for subpart of the surface.
    """
    box, idx = _rg_extrude(fname, voxres, voxshape, origin, directions, extrude=extrude)
    out = np.zeros_like(box)
    rmin, rmax = np.min(idx, axis=0), np.max(idx, axis=0) # Indexes range of surface
    strides = ((rmax - rmin) / div).astype(int)
    cond = lambda x: (mean - std <= x) #& (x <= mean + std)
    #cond = lambda x: mean <= x
    is_inside = lambda l: np.all((inf <= l) & (l < sup), axis=1)
    bound = lambda x, axis: min(x + strides[axis], voxshape[axis])
    for i in range(rmin[0], rmax[0], strides[0]):
        for j in range(rmin[1], rmax[1], strides[1]):
            for k in range(rmin[2], rmax[2], strides[2]):
                # Subdivided seed for more precision
                inf, sup = (i, j, k), (bound(i, 0), bound(j, 1), bound(k, 2))
                sidx = np.unique(idx[is_inside(idx)], axis=0)
                if sidx.size == 0:
                    continue
                seed = vinput[sidx[:,0], sidx[:,1], sidx[:,2]]
                mean, std = np.mean(seed), np.std(seed)
                # Filter annotation to contain voxels close in brightness to surface only
                out[sidx] = np.where(cond(vinput[sidx]), box[sidx], False)
    return out

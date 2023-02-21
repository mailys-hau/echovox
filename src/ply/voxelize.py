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

import numpy as np



def get_affine(directions, spacing):
    # Return affine as defined by VTK
    # Since y is deep axis, we use rows
    dirs = directions / np.linalg.norm(directions, axis=0)
    dirs = dirs * spacing
    origin, rotation = np.zeros((3, 1)), np.array([0, 0, 0, 1])
    return np.vstack([np.hstack([dirs, origin]), rotation])


def get_fname(fname, dname, suffix, fidx=None):
    fidx = f"_{fidx}" if fidx is not None else ''
    return dname.joinpath(f"{fname.stem}{fidx}").with_suffix(suffix)


def to_onehot(labels, skip_classes=[0]):
    classes = [ c for c in np.unique(labels) if c not in skip_classes ]
    onehot = np.zeros([len(classes), *labels.shape], dtype=bool)
    for i, c in enumerate(classes):
        idx = np.argwhere(labels == c)
        onehot[i, idx[:,0], idx[:,1], idx[:, 2]] = 1
    return onehot

def to_labels(onehot, skip_classes=[]):
    labels = np.zeros(onehot.shape[1:], dtype=np.uint8)
    classes = [ c + 1 for c in range(len(onehot)) if c not in skip_classes ]
    #FIXME: Some voxels are both anterior and posterior
    for c in classes:
        idx = np.argwhere(onehot[c - 1] == 1)
        labels[idx[:,0], idx[:,1], idx[:, 2]] = c
    return labels

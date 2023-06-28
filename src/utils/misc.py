import numpy as np



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

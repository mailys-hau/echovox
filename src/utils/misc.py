def get_fname(fname, dname, suffix, n):
    return dname.joinpath(f"{fname.stem}_{n}").with_suffix(suffix)

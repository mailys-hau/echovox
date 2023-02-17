import numpy as np
import trimesh as tm
import trimesh.exchange.ply as tmply



def full_load_ply(file_obj, resolver=None, fix_texture=True, prefer_color=None,
                  *args, **kwargs):
    """ Trimesh's `load_ply` doesn't return normals, we're fixing this """
    # First part is the same as `trimesh.exchange.ply.load_ply`
    elements, is_ascii, image_name = tmply._parse_header(file_obj)
    if is_ascii:
        tmply._ply_ascii(elements, file_obj)
    else:
        tmply._ply_binary(elements, file_obj)
    image = None
    try:
        import PIL.Image
        if image_name is not None:
            data = resolver.get(image_name)
            image = PIL.Image.open(tm.util.wrap_as_stream(data))
    except ImportError:
        tm.util.log.debug("textures require `pip install pillow`")
    except BaseException:
        tm.util.log.warning("unable to load image!", exc_info=True)
    kwargs = tmply._elements_to_kwargs(
            image=image, elements=elements, fix_texture=fix_texture, prefer_color=prefer_color)
    # Check elements for normals and add it to the kwargs
    if "normal" in elements and elements["normal"]["length"]:
        normals = np.column_stack([elements["normal"]["data"][i] for i in "xyz"])
        if not tm.util.is_shape(normals, (-1, 3)):
            raise ValueError("Normals were not (n, 3)!")
        if normals.shape == kwargs["vertices"].shape:
            k = "vertex_normals"
        elif normals.shape[0] == kwargs["faces"].shape[0]:
            k = "face_normals"
        else:
            raise ValueError("Number of normals match neither vertices or faces!")
        kwargs[k] = normals
    return kwargs

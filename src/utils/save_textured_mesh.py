
# From https://github.com/facebookresearch/pytorch3d/pull/332
import pathlib
import warnings
from typing import Optional

import torch
from torchvision import transforms


def save_obj(f, verts, faces, decimal_places: Optional[int] = None, verts_uvs: Optional[list] = None,
             faces_uvs: Optional[list] = None, texture_map: Optional[list] = None):
    """
    Save a mesh to an .obj file.

    Args:
        f: File (or path) to which the mesh should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        decimal_places: Number of decimal places for saving.
                verts_uvs: (V, 2) tensor giving the uv coordinate per vertex.
        faces_uvs: (F, 3) tensor giving the index into verts_uvs for each
                vertex in the face. Padding value is assumed to be -1.
        texture_map:  padded tensor of shape (1, H, W, 3).
    """
    use_texture = verts_uvs is not None and texture_map is not None and faces_uvs is not None
    if use_texture:
        output_path = pathlib.Path(f)
        obj_header = '\nmtllib {0}.mtl\nusemtl mesh\n\n'.format(output_path.stem)

    if len(verts) and not (verts.dim() == 2 and verts.size(1) == 3):
        message = "Argument 'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if len(faces) and not (faces.dim() == 2 and faces.size(1) == 3):
        message = "Argument 'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)


    new_f = False
    if isinstance(f, str):
        new_f = True
        f = open(f, "w")
    elif isinstance(f, pathlib.Path):
        new_f = True
        f = f.open("w")
    try:
        if use_texture:
            f.write(obj_header)
        _save(f, verts, faces, decimal_places, verts_uvs, faces_uvs)
    finally:
        if new_f:
            f.close()
    new_f = False

    if use_texture:
        try:
            # Save texture map to output folder
            transforms.ToPILImage()(texture_map.squeeze().cpu().permute(2, 0, 1)).save(
                output_path.parent / (output_path.stem + '.png'))

            # Create .mtl file linking obj with texture map
            # This potentially could get expanded with different materials etc.
            f_mtl = open(output_path.parent / (output_path.stem + '.mtl'), "w")
            new_f = True

            lines = f'newmtl mesh\n' \
                    f'map_Kd {output_path.stem}.png\n' \
                    f'# Test colors\n' \
                    f'Ka 1.000 1.000 1.000  # white\n' \
                    f'Kd 1.000 1.000 1.000  # white\n' \
                    f'Ks 0.000 0.000 0.000  # black\n' \
                    f'Ns 10.0\n'

            f_mtl.write(lines)
        finally:
            if new_f:
                f_mtl.close()

# TODO (nikhilar) Speed up this function.
def _save(f, verts, faces, decimal_places: Optional[int] = None, verts_uvs: Optional[list] = None,
          faces_uvs: Optional[list] = None) -> None:
    assert not len(verts) or (verts.dim() == 2 and verts.size(1) == 3)
    assert not len(faces) or (faces.dim() == 2 and faces.size(1) == 3)

    if not (len(verts) or len(faces)):
        warnings.warn("Empty 'verts' and 'faces' arguments provided")
        return

    verts, faces = verts.cpu(), faces.cpu()

    lines = ""

    if len(verts):
        if decimal_places is None:
            float_str = "%f"
        else:
            float_str = "%" + ".%df" % decimal_places

        V, D = verts.shape
        for i in range(V):
            vert = [float_str % verts[i, j] for j in range(D)]
            lines += "v %s\n" % " ".join(vert)

    use_texture = (verts_uvs is not None) and (faces_uvs is not None)
    if use_texture:

        verts_uvs, faces_uvs = verts_uvs.cpu(), faces_uvs.cpu()
        assert not len(verts_uvs) or (verts_uvs.dim() == 2 and verts_uvs.size(1) == 2)
        assert not len(faces_uvs) or (faces_uvs.dim() == 2 and faces_uvs.size(1) == 3)

        # Save vertices UVs
        if len(verts_uvs):
            uV, uD = verts_uvs.shape
            for i in range(uV):
                uv = [float_str % verts_uvs[i, j] for j in range(uD)]
                lines += "vt %s\n" % " ".join(uv)

    if torch.any(faces >= verts.shape[0]) or torch.any(faces < 0):
        warnings.warn("Faces have invalid indices")

    if len(faces):
        F, P = faces.shape
        for i in range(F):
            if use_texture:
                # link faces with faces UVs
                face = ["%d/%d" % (faces[i, j] + 1, faces_uvs[i, j] + 1) for j in range(P)]
            else:
                face = ["%d" % (faces[i, j] + 1) for j in range(P)]

            if i + 1 < F:
                lines += "f %s\n" % " ".join(face)

            elif i + 1 == F:
                # No newline at the end of the file.
                lines += "f %s" % " ".join(face)

    f.write(lines)

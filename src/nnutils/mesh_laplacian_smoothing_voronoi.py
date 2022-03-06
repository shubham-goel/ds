import torch


def mesh_laplacian_smoothing(meshes, method: str = "uniform"):
    r"""
    Computes the laplacian smoothing objective for a batch of meshes.
    This function supports three variants of Laplacian smoothing,
    namely with uniform weights("uniform"), with cotangent weights ("cot"),
    and cotangent cuvature ("cotcurv").For more details read [1, 2].

    Args:
        meshes: Meshes object with a batch of meshes.
        method: str specifying the method for the laplacian.
    Returns:
        loss: Average laplacian smoothing loss across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.

    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
    The Laplacian matrix L is a NxN tensor such that LV gives a tensor of vectors:
    for a uniform Laplacian, LuV[i] points to the centroid of its neighboring
    vertices, a cotangent Laplacian LcV[i] is known to be an approximation of
    the surface normal, while the curvature variant LckV[i] scales the normals
    by the discrete mean curvature. For vertex i, assume S[i] is the set of
    neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
    two triangles connecting vertex v_i and its neighboring vertex v_j
    for j in S[i], as seen in the diagram below.

    .. code-block:: python

               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij

        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        For the cotangent variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.

    There is a nice trigonometry identity to compute cotangents. Consider a triangle
    with side lengths A, B, C and angles a, b, c.

    .. code-block:: python

               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C

        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have

        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a

        Putting these together, we get:

        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH


    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.

    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """

    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    uniform_when_unstable = method.endswith('-stable')
    if uniform_when_unstable:
        method = method[:-len('-stable')]

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv", "cotcurv-voronoi"]:
            L, inv_areas = laplacian_cot(meshes,
                                    voronoi=(method=="cotcurv-voronoi"),
                                    uniform_when_unstable=uniform_when_unstable
                                )
            L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            if method == "cot":
                norm_w = torch.where(L_sum>0, 1.0 / L_sum, L_sum)
            else:
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError(f"Method should be one of {{uniform, cot, cotcurv}}. Got {method}")

    if method == "uniform":
        loss = L.mm(verts_packed)
    elif method == "cot":
        loss = L.mm(verts_packed) * norm_w - verts_packed
    elif method in ["cotcurv", "cotcurv-voronoi"]:
        loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
    loss = loss.norm(dim=1)

    loss = loss * weights
    return loss.sum() / N


def laplacian_cot(meshes, voronoi=False, uniform_when_unstable=False, eps=1e-7):
    """
    Returns the Laplacian matrix with cotangent weights and the inverse of the
    face areas.

    Args:
        meshes: Meshes object with a batch of meshes.
    Returns:
        2-element tuple containing
        - **L**: FloatTensor of shape (V,V) for the Laplacian matrix (V = sum(V_n))
           Here, L[i, j] = cot a_ij + cot b_ij iff (i, j) is an edge in meshes.
           See the description above for more clarity.
        - **inv_areas**: FloatTensor of shape (V,) containing the inverse of sum of
           face areas containing each vertex
    """
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    # V = sum(V_n), F = sum(F_n)
    V, F = verts_packed.shape[0], faces_packed.shape[0]

    face_verts = verts_packed[faces_packed]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    # s = 0.5 * (A + B + C)
    # # note that the area can be negative (close to 0) causing nans after sqrt()
    # # we clip it to a small positive value
    # area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()
    # Area as cross product of adjacent sides
    area = (torch.cross(v1-v0, v2-v0, dim=1).norm(dim=1) / 2)
    area_issmall = area<eps
    if uniform_when_unstable:
        area[area_issmall] = float('nan')
    else:
        area[area_issmall] = eps

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    if voronoi:
        # area0 = voronoi-area of vertex0 in that face
        #       = R^2/2 * (sinBcosB+sinCcosC)
        #       = R/4 * (b cosB + c cosC)
        R = A*B*C / (4*area)
        cosa = ((B2+C2-A2)/(2*B*C).clamp(min=1e-12)).clamp(min=-1, max=1)
        cosb = ((-B2+C2+A2)/(2*A*C).clamp(min=1e-12)).clamp(min=-1, max=1)
        cosc = ((B2-C2+A2)/(2*B*A).clamp(min=1e-12)).clamp(min=-1, max=1)
        area0 = (0.25 * R * (B*cosb + C*cosc))
        area1 = (0.25 * R * (A*cosa + C*cosc))
        area2 = (0.25 * R * (B*cosb + A*cosa))
        area_issmall = area_issmall | (area0<eps) | (area1<eps)| (area2<eps)
        if uniform_when_unstable:
            area0[area0<eps] = float('nan')
            area1[area1<eps] = float('nan')
            area2[area2<eps] = float('nan')
        else:
            area0[area0<eps] = eps
            area1[area1<eps] = eps
            area2[area2<eps] = eps

        # darea = (area0+area1+area2-area).abs()
        # print(darea.mean(), (darea/area).mean())
    else:
        area0 = area1 = area2 = area

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces_packed[:, [1, 2, 0]]
    jj = faces_packed[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    idx_flip = torch.stack([jj, ii], dim=0).view(2, F * 3)
    idx = torch.cat([idx,idx_flip], dim=1)
    L_cot = torch.cat([cot.view(-1), cot.view(-1)], dim=0)

    # Vertices adjacent to triangles with small areas may nave 'nan' cotangent
    # Use uniform laplacian for these vertices
    verts_isnan_mask = torch.full_like(verts_packed[:,0], False, dtype=torch.bool)
    verts_isnan_mask[faces_packed[area_issmall,:].view(-1)] = True
    idx_isnan_mask = verts_isnan_mask[idx[0]]
    idx_isnan = idx[:,idx_isnan_mask]
    idx_notnan = idx[:,~idx_isnan_mask]
    cot_isnan = L_cot[idx_isnan_mask]
    cot_notnan = L_cot[~idx_isnan_mask]
    L_notnan = torch.sparse.FloatTensor(idx_notnan, cot_notnan, (V, V))
    L_isnan = torch.sparse.FloatTensor(idx_isnan, torch.ones_like(cot_isnan), (V, V))
    L = L_notnan+L_isnan

    # # Make it symmetric; this means we are also setting
    # # L[v2, v1] = cota
    # # L[v0, v2] = cotb
    # # L[v1, v0] = cotc
    # L += L.t()

    # For each vertex, compute the sum of areas for triangles containing it.
    idx = faces_packed.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=meshes.device)
    val = torch.stack([area0, area1, area2], dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)
    inv_areas[verts_isnan_mask] = inv_areas[~verts_isnan_mask].mean()         # making cotcurv stable. 4 because

    return L, inv_areas

if __name__ == '__main__':
    from pytorch3d.utils import ico_sphere
    for i in range(8,10):
        for method in ['cotcurv', 'cotcurv-stable', 'cotcurv-voronoi', 'cotcurv-voronoi-stable']:
            print(i, method, mesh_laplacian_smoothing(ico_sphere(i), method))
    # for i in range(8,10):
    #     print(mesh_laplacian_smoothing(ico_sphere(i), 'cotcurv-voronoi'))

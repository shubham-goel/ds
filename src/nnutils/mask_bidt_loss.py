import torch
from pytorch3d.ops import knn_points, packed_to_padded, padded_to_packed
from pytorch3d.renderer.mesh.rasterize_meshes import pix_to_non_square_ndc

def create_ndc_xy_image(h: int, w: int, b: int = 1, device: torch.device = torch.device('cpu')):
    """ Create an Bx2xHxW meshgrid containing pytorch3d NDC coordinates at each pixel"""
    x = pix_to_non_square_ndc(torch.linspace(w-1, 0, w, dtype=torch.float, device=device), w, h)   # left is +x
    y = pix_to_non_square_ndc(torch.linspace(h-1, 0, h, dtype=torch.float, device=device), h, w)   # up is +y
    y,x = torch.meshgrid(y,x)
    xy = torch.stack((x,y))[None].expand(b,-1,-1,-1)
    return xy

def maskbidt_loss(mask0, mask1,
        mask0_xy=None, mask1_xy=None,
        margin_pix=0, max_dist=1, K=1,
        reduction='none'
    ) -> torch.Tensor:
    """ mask0,mask1: B,1,H,W mask
        mask0_xy,mask1_xy: B,xy,H,W ndc coordinates
    """
    assert mask0.shape==mask1.shape
    B,_,H,W = mask0.shape
    if (mask0_xy is None) and (mask1_xy is None):
        raise ValueError('Both mask0_xy and mask1_xy are None, calling maskbidt_loss is useless')
    if mask0_xy is None:
        mask0_xy = create_ndc_xy_image(H,W,b=B,device=mask0.device)
    if mask1_xy is None:
        mask1_xy = create_ndc_xy_image(H,W,b=B,device=mask1.device)

    margin = 2*margin_pix/min(H,W)

    # Penalize places where mask0=1 and mask1=0
    loss0 = maskunidt_loss(mask0, mask1, mask0_xy, mask1_xy, margin=margin, max_dist=max_dist, K=K)

    # Penalize places where mask0=0 and mask1=1
    loss1 = maskunidt_loss(mask1, mask0, mask1_xy, mask0_xy, margin=margin, max_dist=max_dist, K=K)

    # Total loss
    loss = loss0+loss1

    if reduction=='none':
        return loss
    elif reduction=='mean':
        return loss.mean()
    elif reduction=='sum':
        return loss.sum()
    else:
        raise ValueError(reduction + " is not valid")

def maskunidt_loss(mask0, mask1, mask0_xy, mask1_xy, margin=0, max_dist=1, K=1, eps=1e-12):
    """ Unidirectional maskdt loss at places where mask0=1 and mask1=0.
        Say mask0[p]=1 and mask1[p]=0. Then loss0[p] = | p-NN(p,mask1_xy[mask1==1]) | is
        distance of p to nearest neighbour in mask1_xy with mask1.

        Returns loss_image: N, 1, H, W
    """
    source_mask = (mask0 * (1-mask1)) > eps
    source_xy, source_xy_nvalid, source_xy_first = get_padded_xy(source_mask, mask0_xy)
    target_xy, target_xy_nvalid, target_xy_first = get_padded_xy(mask1>eps, mask1_xy)
    dists = (       # N,P,K
        knn_points(
            source_xy, target_xy, K = K,
            lengths1 = source_xy_nvalid,
            lengths2 = target_xy_nvalid
        )
        .dists.clamp(min=eps)
        .sqrt()
    )
    loss = (dists - margin).clamp(min=0, max=max_dist).mean(2)
    loss_img = torch.zeros(source_mask.shape, dtype=torch.float, device=source_mask.device)
    loss_img[source_mask] = padded_to_packed(loss, source_xy_first, source_xy_nvalid.sum().item())

    # ensure padded_to_packed is working correctly
    assert(
        (
            mask0_xy.permute(0,2,3,1)[source_mask[:,0]] ==
            padded_to_packed(source_xy, source_xy_first, source_xy_nvalid.sum().item())
        ).all()
    )
    return loss_img

def get_padded_xy(mask_bool, mask_xy):
    """ mask_bool: N,1,H,W
        mask_xy: N,xy,H,W

        Returns mask_xy[mask_bool==True] as a padded tensor
            valid_pts_padded: N,P,xy
            num_valid_pts: N,
    """
    N,_,H,W = mask_bool.shape
    assert mask_xy.shape == (N,2,H,W)

    valid_masks = mask_bool.repeat_interleave(2, dim=1)
    valid_pts_packed = mask_xy.permute(0, 2, 3, 1)[
        valid_masks.permute(0, 2, 3, 1)
    ].reshape(-1, 2)
    first_inds = torch.zeros(N, dtype=torch.int64, device=mask_bool.device)
    num_valid_pts = mask_bool.sum((1, 2, 3))
    first_inds[1:] = num_valid_pts.cumsum(dim=0)[:-1]
    valid_pts_padded = packed_to_padded(
        valid_pts_packed, first_inds, num_valid_pts.max().item()
    )
    return valid_pts_padded, num_valid_pts, first_inds

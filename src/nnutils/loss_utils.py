import torch

def texture_masks(mask_pred, mask_gt, masking='own'):
    """
    mask_pred, mask_gt: B x H x W
    masking: str: describes how to mask gt/pred images before computing loss. Options:
            none (no masking),
            own  (each img multiplied with it's own float mask),
            bool<thresh> (convert each mask to a bool thresholded by thresh)
            share (supervise only where both masks are valid)
            sharebool<thresh> (supervise only where both masks are valid)
    """
    # Masking
    if masking.startswith('share'):
        share = True
        masking = masking[5:]
    else:
        share = False
    if masking == 'none':
        mask_pred = torch.ones_like(mask_pred)
        mask_gt = torch.ones_like(mask_gt)
    elif masking == 'own':
        assert share is False
    elif masking.startswith('bool'):
        _thresh = float(masking[4:])
        mask_pred = mask_pred>_thresh
        mask_gt = mask_gt>_thresh
    else:
        raise ValueError
    if share:
        mask_pred = mask_gt = mask_pred*mask_gt
    return mask_pred, mask_gt

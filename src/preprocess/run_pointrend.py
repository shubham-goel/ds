import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

setup_logger()

coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend


def get_mask_rcnn_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    mask_rcnn_predictor = DefaultPredictor(cfg)
    return mask_rcnn_predictor

def get_pointrend_predictor(detectron2_repo):
    cfg = get_cfg()
    # Add PointRend-specific config
    point_rend.add_pointrend_config(cfg)
    # Load a config from file
    cfg.merge_from_file(f"{detectron2_repo}/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
    predictor = DefaultPredictor(cfg)
    return predictor

def get_category_masks(ims: torch.Tensor, category: str, detectron2_repo: str, debug=False, area_threshold = 0.002):
    """ ims: Nx3xHxW image in [0,1];
        category: coco categry name. Options:
                [person, bicycle, car, motorcycle, airplane,
                bus, train, truck, boat, traffic light,
                fire hydrant, stop sign, parking meter, bench,
                bird, cat, dog, horse, sheep, cow, elephant,
                bear, zebra, giraffe, backpack, umbrella,
                handbag, tie, suitcase, frisbee, skis,
                snowboard, sports ball, kite, baseball bat,
                baseball glove, skateboard, surfboard, tennis racket,
                bottle, wine glass, cup, fork, knife, spoon,
                bowl, banana, apple, sandwich, orange,
                broccoli, carrot, hot dog, pizza, donut,
                cake, chair, couch, potted plant, bed,
                dining table, toilet, tv, laptop, mouse,
                remote, keyboard, cell phone, microwave,
                oven, toaster, sink, refrigerator, book,
                clock, vase, scissors, teddy bear, hair drier,
                toothbrush]
        Returns masks: Nx1xHxW
    """

    category_id = coco_metadata.thing_classes.index(category)
    predictor = get_pointrend_predictor(detectron2_repo)

    # Convert images to BGR numpy images
    device = ims.device
    N,_,H,W = ims.shape
    ims = (ims*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
    ims = ims[:,:,:,::-1]
    assert(ims.shape[3]==3)

    masks = []
    for i,im in enumerate(ims):
        instances = predictor(im)['instances']
        useful_instances = instances.pred_classes==category_id   # Only the classes we want
        useful_instances &= instances.pred_masks.sum(dim=(1,2)) > (np.prod(instances.image_size) * area_threshold) # Only big-enough instances
        mask = instances.pred_masks[useful_instances].any(dim=0)
        masks.append(mask)

        if debug:
            v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            point_rend_result = v.draw_instance_predictions(instances.to("cpu")).get_image()
            viz_img = point_rend_result[:, :, ::-1]

            vizdir = Path('/private/home/shubhamgoel/code/3Dify/sandbox/debug_pointrend/')
            vizdir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vizdir / f'{i}_mask.jpg'), (mask.float()*255).cpu().numpy().astype(np.uint8))
            cv2.imwrite(str(vizdir / f'{i}_viz.jpg'), viz_img)

    masks = torch.stack(masks)[:,None,:,:].to(device)
    assert(masks.shape == (N,1,H,W))

    return masks

def run_maskrcnn_pointrend_on_imgs(img_paths: List[Path], vizdir: Path, detectron2_repo: str):
    # Used for visualization and debuugging

    mask_rcnn_predictor = get_mask_rcnn_predictor()
    predictor = get_pointrend_predictor(detectron2_repo)

    for impath in img_paths:
        print('Running on', impath)
        im = cv2.imread(str(impath))
        mask_rcnn_outputs = mask_rcnn_predictor(im)
        outputs = predictor(im)

        # Show and compare two predictions:
        v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        mask_rcnn_result = v.draw_instance_predictions(mask_rcnn_outputs["instances"].to("cpu")).get_image()
        v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        # print("Mask R-CNN with PointRend (top)     vs.     Default Mask R-CNN (bottom)")
        viz_img = np.concatenate((point_rend_result, mask_rcnn_result), axis=0)[:, :, ::-1]

        # cv2_imshow(viz_img)


        vizdir.mkdir(parents=True, exist_ok=True)
        outpath = vizdir / f'{impath.name}_vizout.jpg'
        cv2.imwrite(str(outpath), viz_img)

if __name__ == '__main__':
    detectron2_repo = os.environ['SHARED_HOME'] + '/code/detectron2/'
    in_dir = Path('/shared/shubham/data/tankandtemples/intermediate/Horse/images/')
    out_dir = Path('debug/pointrend/viz/Horse/').absolute()

    print(in_dir)
    print(out_dir)

    img_paths = sorted(list(in_dir.glob('*.jpg')))
    run_maskrcnn_pointrend_on_imgs(img_paths, out_dir, detectron2_repo = detectron2_repo)


This document describes how to run our implementation of different baselines in the paper.

## NeRF
Our NeRF implementation builds on [Pytorch3D's NeRF implementation](https://github.com/facebookresearch/pytorch3d/tree/main/projects/nerf). Specifically, we modified the NeRF implementation to 1) Sample 50% points from inside the GT object mask, and 2) Supervise the rendered mask.

To run NeRF, simply run the `src.exp.nerf` module instead of `src.exp.ds` with the same data config. For example, the following command runs NeRF on the GSO dataset:
```bash
python -m src.exp.nerf --config-name=nerf \
data/source=google data.cam_noise.std_rot=30 data.num_views=8 \
data.source.root_dir=$GSO_DATA_DIR data.source.asin=Schleich_Bald_Eagle \
exp_name=gso_nerf
```

Similar to DS, you may futher evaluate and visualize a trained NeRF model using:
```bash
python -m src.eval.evaluate_nerf asin=eagle exp_dir='models_NeRF/eagle/' out_dir='viz_NeRF/'
```

## NeRF-opt
The NeRF-opt baseline additionally optimizes camera pose parameters while training a NeRF. The NeRF-opt baseline can be run using the `--config-name=nerf_opt` config. For example:
```bash
python -m src.exp.nerf --config-name=nerf_opt \
data/source=google data.cam_noise.std_rot=30 data.num_views=8 \
data.source.root_dir=$GSO_DATA_DIR data.source.asin=Schleich_Bald_Eagle \
exp_name=gso_nerfopt
```

## COLMAP
Our COLMAP experiments use [FreeViewSynthesis's Python API](https://raw.githubusercontent.com/intel-isl/FreeViewSynthesis/33a31ee214a77a2fa074d3a10cedc09803ec2ceb/co/colmap.py). First, set the COLMAP binary and library paths in [configs/colmap.yaml](../configs/colmap.yaml). Then to run COLMAP, simply run the `src.exp.colmap` module instead of `src.exp.ds` with the same data config. For example, the following command runs COLMAP (with GT cameras) on the GSO dataset:
```bash
python -m src.exp.colmap --config-name=colmap \
    data/source=google data.num_views=8 \
    data.source.root_dir=$GSO_DATA_DIR data.source.asin=Schleich_Bald_Eagle \
    exp_name=gso_colmap
```

To evaluate COLMAP reconstructed point clouds, see [src/eval/evaluate_colmap.py](../src/eval/evaluate_colmap.py).

## IDR
We offer a script to convert data to format that's friendly to [IDR](https://github.com/lioryariv/idr). You may also use [our fork of IDR](https://github.com/shubham-goel/idr/tree/ds_eval) which enables dataloading for GSO and Tanks and Temples.

For example, to convert data for `Schleich_Bald_Eagle` (from GSO) with 8 views and 30 degrees camera noise:
```bash
python -m src.preprocess.data_to_idr_format \
data/source=google data.cam_noise.std_rot=30 data.num_views=8 \
data.source.root_dir=$GSO_DATA_DIR data.source.asin=Schleich_Bald_Eagle \
out_dir=idr_data/
```
This saves the images, masks and cameras (both GT and noisy) to `idr_data/google/r30t0h0/v8/Schleich_Bald_Eagle/`. You can now preprocess the cameras, run IDR, and export the cameras and shape as shown in [this file](https://github.com/shubham-goel/idr/blob/ds_eval/code/run_gso.sh).

To evaluate IDR meshes and cameras consistently like DS, see [src/eval/evaluate_idr.py](../src/eval/evaluate_idr.py). After running IDR on all 50 instances in [all_benchmark_google_asins.txt](../all_benchmark_google_asins.txt), you may compute metrics for IDR as follows:
```bash
python -m src.eval.evaluate_idr aggregate=True
```

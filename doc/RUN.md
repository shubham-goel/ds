This document describes how to run DS on different datasets.
## Demo data
### Running on custom data
You may run DS on custom data. Download the sample data from [here](https://people.eecs.berkeley.edu/~shubham-goel/projects/ds/example_data.zip) and unzip it. This should create `example_data` in the root DS directory. The structure of this data is discussed later. Now run DS on this data: you should specify number of views to use for training, and may optionally add noise to the camera extrinsics.
```bash
python -m src.exp.ds --config-name=ds \
data/source=demo data.source.asin=eagle \
data.num_views=12 data.cam_noise.std_rot=30 \
exp_name=demo
```

### Visualizing a pretrained model
Download a pretrained DS model from [here](https://people.eecs.berkeley.edu/~shubham-goel/projects/ds/models.zip) and unzip it to `models/`. We can now extract the mesh, and visualize novel views using the evaluation class in [src/eval/evaluate_ds.py](../src/eval/evaluate_ds.py). For example, the following script extracts visualizations to the `viz/` directory.
```bash
python -m src.eval.evaluate_ds asin=eagle exp_dir='models/eagle/' out_dir='viz/'
```

<p align="center">
  <img width="600" src="./eagle.gif"/>
</p>

### Data Format
The structure of the example data is as folows:
```
└── eagle
    ├── meshes (optional)
    │   ├── model.mtl
    │   ├── model.obj
    │   └── texture.png
    ├── r_0.png
    ├── r_1.png
    ...
    ├── r_19.png
    └── transforms.json
```
`transforms.json` contains filenames, camera extrinsics and intrinsics for each image. `file_path` indicates the corresponding image. The `tranform_matrix` field is the 4x4 camera to world transformation matrix. It follows the blender coordinate system: Y up, X right, Z behind. The `camera` field defines the camera instrinsics: `angle_x` and `angle_y` are the horizontal and vertical field of view in radians. Our current implementation assumes that the principal point is at the image centre.

## Google's Scanned Objects dataset
This section describes how to download, render and run on [Google's Scanned Objects dataset](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects). Instead of downloading and rendering the entire dataset, you may choose to only download the subset of 50 randomly chosen instances we evaluated on -- as listed in [all_benchmark_google_asins.txt](../all_benchmark_google_asins.txt).

### Download the GSO dataset
Let `DS_DIR` be the root directory of the DS codebase. Set `GSO_DATA_DIR` to where you wish to download the GSO dataset.
```bash
mkdir $GSO_DATA_DIR && cd $GSO_DATA_DIR
python $DS_DIR/src/preprocess/download-google-models.py # Downloads the GSO dataset to the current directory
for f in *.zip; do mkdir ${f%.zip}; unzip "$f" -d "${f%.zip}/1/"; done      # Extract all
```

### Render the GSO dataset
Render all the instances using [blender](https://www.blender.org/download/releases/2-82/) (we used Blender 2.82). First, set the `GOOGLE_DATASET_DIR` in [src/preprocess/blender-render-google.py](../src/preprocess/blender-render-google.py) to `$GSO_DATA_DIR`. Then, render all instances as follows. Rendered images and metadata are saved to a `renders_env/` subdirectory for each instance directory.
```bash
blender -b --python $DS_DIR/src/preprocess/blender-render-google.py -- .*
```

### Running DS on the GSO dataset
You may now run DS on any instance with the settings of your choice. For example, 8 views of "Schleich_Bald_Eagle" with 30 degrees std Gaussian noise in camera rotation):
```bash
python -m src.exp.ds --config-name=ds \
data/source=google data.cam_noise.std_rot=30 data.num_views=8 \
data.source.root_dir=$GSO_DATA_DIR data.source.asin=Schleich_Bald_Eagle \
exp_name=gso_ds
```

## Tanks and Temples dataset
To run DS on the [Tanks and Temples dataset](https://www.tanksandtemples.org/), you may download the dataset from [the official source](https://www.tanksandtemples.org/download/) or [the MVSNet repo](https://github.com/YoYo000/MVSNet). 

Modify the [configs/data/source/tanks_and_temples.yaml](../configs/data/source/tanks_and_temples.yaml) config to point to the downloaded data. Now run DS as follows:
```bash
python -m src.exp.ds --config-name=ds \
    data/source=tanks_and_temples data.num_views=15 data.image_size=[306,544] \
    data.source.asin=Caterpillar \
    exp_name=t2_ds
```

To run on the `Horse` scene where GT pointclouds are not available for rendering to masks, we can get masks from MaskRCNN+PointRend. First, clone and install detecron2==0.5. Then run:
```bash
python -m src.exp.ds --config-name=ds_t2_pointrend \
    data/source=tanks_and_temples data.num_views=15 data.image_size=[306,544] \
    data.source.asin=Horse data.source.mask_coco_category=horse \
    data.source.detectron2_repo_path=/path/to/detecron2_v0.5/ \
    exp_name=t2_ds
```
The reconstructed mesh and texture can be visualizes like GSO using [src/eval/evaluate_ds.py](../src/eval/evaluate_ds.py).

## Replicating paper results
_Warning: Replicating all DS numbers in the paper requires a total compute of ~300 GPU days._
To replicate the numbers in Fig. 3 of the paper, first run DS  and NeRF-opt (see [BASELINES.md](BASELINES.md)) on all 50 instances listed in [all_benchmark_google_asins.txt](../all_benchmark_google_asins.txt) with `data.cam_noise.std_rot=10,20,30` and `data.num_views=4,6,8,12`. Then compute and extract metrics as follows:
```bash
# Copmute metrics with different alignment settings
python -m src.eval.aggregate_results reaggregate=True align_fine=none
python -m src.eval.aggregate_results reaggregate=True align_fine=icp_p2g_noscale_centered
python -m src.eval.aggregate_results reaggregate=True align_fine=icp_g2p_noscale_centered

# Plot the metrics, by choosing the best of 3 alignments above
python -m src.eval.plot_results
```

#### DS Ablations
This section describes how to run the `DS-naive` and `DS-notex` ablations.

`DS-notex` doesn't use any texture information:
```bash
python -m src.exp.ds --config-name=ds_notex \
data/source=google data.cam_noise.std_rot=30 data.num_views=8 \
data.source.root_dir=$GSO_DATA_DIR data.source.asin=US_Army_Stash_Lunch_Bag \
exp_name=gso_ds_notex
```
`DS-naive` optimizes a UV texture map instead of transferring it:
```bash
python -m src.exp.ds --config-name=ds_naive \
data/source=google data.cam_noise.std_rot=30 data.num_views=8 \
data.source.root_dir=$GSO_DATA_DIR data.source.asin=US_Army_Stash_Lunch_Bag \
exp_name=gso_ds_naive
```

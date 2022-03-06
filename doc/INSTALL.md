## Installing DS
Here are DS's major dependencies:
1. [Pytorch3D](https://github.com/facebookresearch/pytorch3d)
1. [Binvox](https://www.patrickmin.com/binvox/)
1. [Hydra](https://hydra.cc/)

To run DS, first download the [Binvox](https://www.patrickmin.com/binvox/) binary to the root DS directory.  Then you may create a conda virtual environment for DS (on Linux), using the [environment.yaml](../environment.yaml) file:
```bash
conda env create -f environment.yaml
```

Alternatively, follow these steps:
```bash
conda create -y -n ds \
-c pytorch -c pytorch3d -c conda-forge -c fvcore -c iopath -c bottler \
python=3.8 \
pytorch setuptools=59.5.0 \
cudatoolkit=10.2 \
fvcore iopath nvidiacub pytorch3d=0.5.0

conda activate ds

conda install -y -c conda-forge -c open3d-admin \
open3d \
scikit-image  imageio \
configargparse \
dotmap moviepy ffmpeg \
tensorboard visdom plotly matplotlib

# Pip stuffs
pip install opencv-python opencv-contrib-python
pip install hydra-submitit-launcher==1.1.6 --upgrade
pip install hydra-core==1.1.1
pip install hydra-colorlog
pip install PyMCubes --upgrade
pip install lpips
```

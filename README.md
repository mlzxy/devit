# Detect Every Thing with Few Examples

[Paper]()

![](images/pipeline.jpg)

We present DE-ViT, an open-set object detector in this repository.
In contrast to the popular open-vocabulary approach, we follow the few-shot formulation to represent each category with few support images rather than language. Our results shows potential for using images as category representation. 
DE-ViT establishes new state-of-the-art on open-vocabulary, few-shot, and one-shot object detection benchmark with COCO and LVIS.

## Installation

```bash
git clone https://github.com/mlzxy/devit.git
conda create -n devit  python=3.9 
conda activate devit
pip install -r devit/requirements.txt
pip install -e ./devit
```

Next, check [Downloads.md](Downloads.md) for instructions to setup datasets and model checkpoints.

## Running Scripts

<!-- python3 ./tools/extract_instance_prototypes.py   --dataset fs_coco17_train_base    --model vits14


 python3 ./tools/run_sinkhorn_cluster.py  --inp  ./one_shot_s1.crop_paste.pkl   --epochs 10    --momentum 0.002   --num_prototypes 10
 -->


## Demo

![](demo/output/ycb.out.jpg)

```

```

## Evaluation 

```bash

```

## Training 

```bash

```

## RPN Training

```
```

Check [Tools.md](Tools.md) for intructions to build prototype and prepare weights.

## Acknowledgement


This repository was built on top of Detectron2, DINOv2, RegionCLIP. We thank the effort from our community.


<!-- 
## Citation

```
```
-->





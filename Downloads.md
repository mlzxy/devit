# Download

All datasets and model checkpoints are stored in 
https://drive.google.com/drive/folders/1b3anUR2Gloh7XpvevPCuxWeYCvQJ7Pfz.

You can download all of them using [gdown](https://github.com/wkentaro/gdown)

```bash
cd devit
pip install gdown
gdown 1b3anUR2Gloh7XpvevPCuxWeYCvQJ7Pfz --folder
# will download a folder called `release`
mv release/weights release/datasets .
rm -rf release
```



## Datasets Preparation (folder `datasets`)

```bash
├── [   0]  coco/
│   └── [   0]  annotations/ # different COCO subsamples, according to benchmark specs
│       ├── [1.5M]  coco_2017_novel_oneshot_s1_r100.json  # sampled images of novel category for one-shot split-1, only used for eval during training (not full eval)
│       ├── [763K]  coco_2017_novel_oneshot_s1_r50.json
│       ├── [1.4M]  coco_2017_novel_oneshot_s2_r100.json
│       ├── [746K]  coco_2017_novel_oneshot_s2_r50.json
│       ├── [1.5M]  coco_2017_novel_oneshot_s3_r100.json
│       ├── [758K]  coco_2017_novel_oneshot_s3_r50.json
│       ├── [1.5M]  coco_2017_novel_oneshot_s4_r100.json
│       ├── [747K]  coco_2017_novel_oneshot_s4_r50.json
│       ├── [233M]  coco_2017_train_oneshot_s1.json # base category for one-shot split-1
│       ├── [383M]  coco_2017_train_oneshot_s2.json
│       ├── [368M]  coco_2017_train_oneshot_s3.json
│       ├── [365M]  coco_2017_train_oneshot_s4.json
│       ├── [ 10M]  coco_2017_val_oneshot_s1.json # base category for one-shot split-1, at val set
│       ├── [ 16M]  coco_2017_val_oneshot_s2.json
│       ├── [ 16M]  coco_2017_val_oneshot_s3.json
│       ├── [ 16M]  coco_2017_val_oneshot_s4.json
│       ├── [121M]  fs_coco14_base_train.json 
│       ├── [ 58M]  fs_coco14_base_val.json
│       ├── [467M]  ovd_ins_train2017_all.json # From RegionCLIP, OVD COCO
│       ├── [406M]  ovd_ins_train2017_b.json
│       ├── [ 72M]  ovd_ins_train2017_t.json
│       ├── [ 20M]  ovd_ins_val2017_all.json
│       ├── [ 17M]  ovd_ins_val2017_b.json
│       └── [3.0M]  ovd_ins_val2017_t.json
├── [   0]  lvis/  # the following two are not used in training. LVIS is splited on-demand during data loading
|   |
│   ├── [1.1G]  lvis_v1_known_train.pkl # common+frequent
│   └── [5.2M]  lvis_v1_novel_train.pkl # rare
|
├── [797K]  cocosplit2017.tar.gz # samples for novel categories for COCO-2017, used in OVD experiments
└── [159M]  cocosplit.tar.gz  # all few-shot COCO-2014 base/novel splits sampled by previous work
```

Instructions: 

```bash
mv datasets/coco/annotations/* $DETECTRON2_DATASETS/coco/annotations/
mv datasets/lvis/* $DETECTRON2_DATASETS/lvis
tar xvf datasets/cocosplit2017.tar.gz  -C  $DETECTRON2_DATASETS
tar xvf datasets/cocosplit.tar.gz -C $DETECTRON2_DATASETS
```

Note that you need to first setup COCO14/17 and LVIS in your detectron2 datasets folder. 


## Checkpoint Structures (folder `weights`)

```bash
├── [   0]  initial/ # initial weights, including pre-built prototypes
│   ├── [   0]  background/ # background prototypes
│   ├── [   0]  demo/ # prototypes for YCB
│   ├── [   0]  DINOv2/
│   ├── [   0]  few-shot/
│   ├── [   0]  oneshot/
│   ├── [   0]  open-vocabulary/
│   └── [   0]  rpn/
└── [   0]  trained/
    ├── [   0]  few-shot/
    ├── [   0]  one-shot/
    └── [   0]  open-vocabulary/
```
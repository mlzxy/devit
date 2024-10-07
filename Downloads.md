# Download

All datasets and model checkpoints are stored in https://rutgers.box.com/s/2lco6ab66pn3ufq6rh4gmyfzg9vfkm23. To download all files efficiently, there is a large `all_datasets_and_weights.tar.gz` file in that box link that includes all datasets and weights. 


```bash
cd devit

# download the `all_datasets_and_weights.tar.gz` from box

tar xvf all_datasets_and_weights.tar.gz # will create a folder called `release`
mv release/weights release/datasets .
rm -rf release all_datasets_and_weights.tar.gz
```

>use `rclone` with more instructions from [issue#7](https://github.com/mlzxy/devit/issues/7). 


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
unzip datasets/vocsplit.zip -d $DETECTRON2_DATASETS
tar xvf datasets/cocosplit2017.tar.gz  -C  $DETECTRON2_DATASETS
tar xvf datasets/cocosplit.tar.gz -C $DETECTRON2_DATASETS
```

Note that you need to first setup COCO14/17, Pascal VOC, and LVIS in your detectron2 datasets folder. 


## Checkpoint Structures (folder `weights`)

```bash
├── [   0]  initial/ # initial weights, including pre-built prototypes
│   ├── [   0]  background/ # background prototypes
│   │   ├── [1.6M]  background_prototypes.vitb14.pth
│   │   ├── [2.1M]  background_prototypes.vitl14.pth
│   │   └── [796K]  background_prototypes.vits14.pth
│   ├── [   0]  DINOv2/ # vit weights
│   │   ├── [330M]  vitb14.pth
│   │   ├── [1.1G]  vitl14.pth
│   │   └── [ 84M]  vits14.pth
│   ├── [   0]  few-shot/  # class prototypes and initial model weights for few-shot COCO14
│   │   ├── [   0]  prototypes/
│   │   ├── [399M]  vitb+rpn.pth
│   │   ├── [1.2G]  vitl+rpn.pth
│   │   └── [153M]  vits+rpn.pth
│   ├── [   0]  oneshot/ # similar to few-shot, but for one-shot
│   │   ├── [   0]  prototypes_for_train/
│   │   ├── [ 79M]  ref_coco17.vitl14.pth # all instance prototypes of coco17 val set, used for full eval
│   │   ├── [1.2G]  vitl+rpn.split1.pth
│   │   ├── [1.2G]  vitl+rpn.split2.pth
│   │   ├── [1.2G]  vitl+rpn.split3.pth
│   │   └── [1.2G]  vitl+rpn.split4.pth
│   ├── [   0]  open-vocabulary/ # for ovd
│   │   ├── [   0]  prototypes/
│   │   ├── [438M]  vitb+rpn_lvis.pth
│   │   ├── [399M]  vitb+rpn.pth
│   │   ├── [1.2G]  vitl+rpn_lvis.pth
│   │   ├── [1.2G]  vitl+rpn.pth
│   │   ├── [191M]  vits+rpn_lvis.pth
│   │   └── [153M]  vits+rpn.pth
│   └── [   0]  rpn/ # pretrained RPN for each setting
│       ├── [   0]  few-shot-coco14/
│       ├── [   0]  one-shot-split1/
│       ├── [   0]  one-shot-split2/
│       ├── [   0]  one-shot-split3/
│       ├── [   0]  one-shot-split4/
│       ├── [   0]  open-vocabulary-coco17/
│       └── [   0]  open-vocabulary-lvis/
└── [   0]  trained/
    ├── [   0]  few-shot/ # few-shot model, with train/eval log
    │   ├── [428M]  vitb_0089999.pth
    │   ├── [602K]  vitb.eval-log-30-shot.txt
    │   ├── [601K]  vitb.eval-log-5-shot.txt
    │   ├── [4.1M]  vitb.train-log-10-shot.txt
    │   ├── [1.2G]  vitl_0089999.pth
    │   ├── [653K]  vitl.eval-log-30-shot.txt
    │   ├── [622K]  vitl.eval-log-5-shot.txt
    │   ├── [4.3M]  vitl.train-log-10-shot.txt
    │   ├── [181M]  vits_0089999.pth
    │   ├── [508K]  vits.eval-10-shot-box.txt
    │   ├── [508K]  vits.eval-30-shot-box.txt
    │   ├── [557K]  vits.eval-30-shot.txt
    │   ├── [519K]  vits.eval-5-shot-box.txt
    │   ├── [565K]  vits.eval-5-shot.txt
    │   └── [4.2M]  vits.train-log-10-shot.txt
    ├── [   0]  one-shot/ # one-shot model, with train and eval log
    │   ├── [1.0M]  log-eval.split1.txt
    │   ├── [772K]  log-eval.split2.txt
    │   ├── [840K]  log-eval.split3.txt
    │   ├── [764K]  log-eval.split4.txt
    │   ├── [4.1M]  log-train.split1.txt
    │   ├── [4.0M]  log-train.split2.txt
    │   ├── [7.9M]  log-train.split3.txt
    │   ├── [4.0M]  log-train.split4.txt
    │   ├── [1.2G]  vitl_0049999.split2.pth
    │   ├── [1.2G]  vitl_0064999.split3.pth
    │   ├── [1.2G]  vitl_0074999.split1.pth
    │   └── [1.2G]  vitl_0084999.split4.pth
    └── [   0]  open-vocabulary/
        ├── [   0]  coco/ # ovd model, with train/eval log on COCO
        │   ├── [428M]  vitb_0079999.pth
        │   ├── [ 31K]  vitb.eval.log.txt
        │   ├── [4.0M]  vitb.train.log.txt
        │   ├── [1.2G]  vitl_0064999.pth
        │   ├── [ 33K]  vitl.eval.log.txt
        │   ├── [4.0M]  vitl.train.log.txt
        │   ├── [181M]  vits_0034999.pth
        │   ├── [ 30K]  vits.eval.log.txt
        │   └── [4.0M]  vits.train.log.txt
        └── [   0]  lvis/ # ovd model, with train/eval log on LVIS
            ├── [520M]  vitb_0059999.pth
            ├── [4.3M]  vitb.train-box.log.txt
            ├── [3.8M]  vitb.train-mask.log.txt
            ├── [1.3G]  vitl_0069999.pth
            ├── [3.2K]  vitl.eval.log.txt
            ├── [269M]  vits_0059999.pth
            ├── [2.9M]  vits.train-box.log.txt
            └── [3.7M]  vits.train-mask.log.txt

```

Note that 

- The periodic eval results in OVD COCO training log files are lower than reported due to a bug at the time. Please check the corresponding eval log.

- The lvis training log is splited into box and mask because our initial version only has box prediction and we design and train the mask head separately afterwards (with the same number of iterations). This is fine because mask branch is completely independent from other branches. Parts of the training log of LVIS vitl mask branch is missing due to human errors (check the eval log). 

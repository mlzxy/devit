# Build Prototypes

```bash
# Generate instance prototypes for base category of OVD COCO
python3 ./tools/extract_instance_prototypes.py   --dataset fs_coco17_train_base  --model vits14
# will produce a file `fs_coco17_train_base.vits14.pkl`

# Generate instance prototypes for novel category of OVD COCO
python3 ./tools/extract_instance_prototypes.py   --dataset fs_coco17_support_novel_30shot  --model vits14  --epochs 60
# will produce a file `fs_coco17_support_novel_30shot.vits14.pkl`
# back then, augmentation is used, so the `epoch` parameter exists. But we find augmentation is not a critical element.
# So we remove it. But in order to keep consistency, we still add some epochs.

# Generate class prototypes through clustering 
python3 ./tools/run_sinkhorn_cluster.py  --inp  ./fs_coco17_train_base.vits14.pkl  --epochs 10   --momentum 0.002   --num_prototypes 10
# will produce a file `fs_coco17_train_base.vits14.p10.sk.pkl`

python3 ./tools/run_sinkhorn_cluster.py  --inp  ./fs_coco17_support_novel_30shot.vits14.pkl  --epochs 30 --momentum 0.002    --num_prototypes 10
# will produce a file `fs_coco17_support_novel_30shot.vits14.p10.sk.pkl`
```

Then, set `DE.CLASS_PROTOTYPES` to `fs_coco17_train_base.vits14.p10.sk.pkl,fs_coco17_support_novel_30shot.vits14.p10.sk.pkl` to use the above generated prototypes. The stuff classes of `coco_2017_train_panoptic_stuffonly` dataset are used to extract background prototypes with the same procedure.


A list of base / novel dataset pairs: 

- LVIS `lvis_v1_known_train / lvis_v1_novel_train` (note that `lvis_v1_train` are used in config because they will be splited during loading, check <detectron2/data/datasets/lvis.py>.

- Few-Shot COCO 14 `fs_coco14_base_train / fs_coco_trainval_novel_10shot`.

- OVD COCO 17 `coco_2017_ovd_b_train / fs_coco17_support_novel_30shot`. `fs_coco17_train_base` is an alias for `coco_2017_ovd_b_train`.

- One-Shot COCO split 1 `coco_2017_train_oneshot_s1 / coco_2017_novel_oneshot_s1_r50` (r50 means reservoir 50 samples, we randomly pick 50 samples for eval during training)


# Combine RPN weights with ViT weights

Since our model relies on ViT and RPN, we need to combine them into a single checkpoint. All initial models with names like `vits+rpn.pth` are pre-combined. The tool used is [tools/combine_vit_rpn_weights.py](tools/combine_vit_rpn_weights.py).
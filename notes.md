python3 ./tools/extract_instance_prototypes.py   --dataset fs_coco17_train_base    --model vits14


 python3 ./tools/run_sinkhorn_cluster.py  --inp  ./one_shot_s1.crop_paste.pkl   --epochs 10    --momentum 0.002   --num_prototypes 10
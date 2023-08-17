task="${task:-ovd}" # ovd, fsod, osod
vit="${vit:-l}" # s, b, l
dataset="${dataset:-coco}" # coco, lvis
shot="${shot:-10}"
split="${split:-1}"
num_gpus="${num_gpus:-`nvidia-smi -L | wc -l`}"


case $task in

    
    ovd)

    if [[ "$dataset" == "coco" ]]
    then
    echo ""
    else
    echo ""
    fi

    ;;


    fsod)

    ;;


    osod)
    python3 tools/train_net.py \
        --num-gpus $num_gpus  --eval-only \
        --config-file configs/one-shot/s${split}_vitL.yaml \
        MODEL.WEIGHTS  `ls weights/trained/one-shot/vit${vit}_*.split${split}.pth` \
        DE.OFFLINE_RPN_CONFIG  configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
        OUTPUT_DIR  output/one-shot/split${split}/vit${vit}/ \
        DE.ONE_SHOT_MODE  True
        ;;

    *)
        echo "skip"
        ;;
esac


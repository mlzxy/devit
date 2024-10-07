task="${task:-ovd}" # ovd, fsod, osod
vit="${vit:-l}" # s, b, l
dataset="${dataset:-coco}" # coco, lvis
shot="${shot:-10}"
split="${split:-1}"
num_gpus="${num_gpus:-`nvidia-smi -L | wc -l`}"

echo "task=$task, vit=$vit, dataset=$dataset, shot=$shot, split=$split, num_gpus=$num_gpus"

case $task in

    ovd)
    if [[ "$dataset" == "coco" ]]
    then
        python3 tools/train_net.py    --num-gpus $num_gpus --eval-only \
            --config-file configs/open-vocabulary/coco/vit${vit}.yaml \
            MODEL.WEIGHTS `ls weights/trained/open-vocabulary/coco/vit${vit}_*.pth | head -n 1` \
            DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
            OUTPUT_DIR output/eval/open-vocabulary/coco/vit${vit}/ $@
    else
        python3 tools/train_net.py    --num-gpus $num_gpus  --eval-only \
            --config-file  configs/open-vocabulary/lvis/vit${vit}.yaml \
            MODEL.WEIGHTS  `ls weights/trained/open-vocabulary/lvis/vit${vit}_*.pth | head -n 1` \
            DE.OFFLINE_RPN_CONFIG  configs/RPN/mask_rcnn_R_50_FPN_1x.yaml \
            OUTPUT_DIR output/eval/open-vocabulary/lvis/vit${vit}/ $@
    fi
    ;;

    fsod)
    if [[ "$dataset" == "coco" ]]
    then 
        python3 tools/train_net.py --num-gpus $num_gpus --eval-only \
            --config-file configs/few-shot/vit${vit}_shot${shot}.yaml  \
            MODEL.WEIGHTS `ls weights/trained/few-shot/vit${vit}_*.pth | head -n 1`  \
            DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
            OUTPUT_DIR output/eval/few-shot/shot-${shot}/vit${vit}/  $@
    else
        python3 tools/train_net.py --num-gpus $num_gpus --eval-only \
            --config-file configs/few-shot-voc/${shot}shot/vit${vit}_${split}s.yaml  \
            MODEL.WEIGHTS `ls weights/trained/few-shot-voc/${split}/vit${vit}_*.pth | head -n 1`  \
            DE.OFFLINE_RPN_CONFIG configs/VOC_RPN/faster_rcnn_R_50_C4.few_shot_s1.yaml \
            OUTPUT_DIR output/eval/few-shot-voc/${shot}shot/${split}/vit${vit}/  $@
    fi
    ;;

    osod)
        python3 tools/train_net.py \
            --num-gpus $num_gpus  --eval-only \
            --config-file configs/one-shot/split${split}_vit${vit}.yaml \
            MODEL.WEIGHTS  `ls weights/trained/one-shot/vit${vit}_*.split${split}.pth | head -n 1` \
            DE.OFFLINE_RPN_CONFIG  configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
            OUTPUT_DIR  output/eval/one-shot/split${split}/vit${vit}/ \
            DE.ONE_SHOT_MODE  True $@
            ;;
    *)
        echo "skip"
        ;;
esac


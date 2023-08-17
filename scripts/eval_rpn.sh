# 可能需要在 train_net.py 里加一个 online 组合 weight，生成临时 checkpoint 逻辑，否则要生成太多 initial checkpoint 了
# 

set -o xtrace

case $1 in

        ovd)
        if [[ "$2" == "base" ]]
        then
            python3 tools/train_net.py --eval-only \
                    --num-gpus 4  \
                    --config-file  configs/COCO-Detection/rpn_R_50_C4_1x.yaml \
                    OUTPUT_DIR  output/rpn/open-vocabulary-coco17/base \
                    MODEL.WEIGHTS  weights/initial/rpn/open-vocabulary-coco17/rpn_coco_48.pth \
                    DATASETS.TEST '("coco_2017_ovd_b_test",)'
        else
            python3 tools/train_net.py --eval-only \
                    --num-gpus 4  \
                    --config-file  configs/COCO-Detection/rpn_R_50_C4_1x.yaml \
                    OUTPUT_DIR  output/rpn/open-vocabulary-coco17 \
                    MODEL.WEIGHTS weights/initial/rpn/open-vocabulary-coco17/rpn_coco_48.pth \
                    DATASETS.TEST '("coco_2017_ovd_all_test",)'
        fi
        ;;

        os1)
        if [[ "$2" == "base" ]]
        then
            python3 tools/train_net.py --eval-only \
                    --num-gpus 4  \
                    --config-file  configs/COCO-Detection/rpn_R_50_C4_1x.yaml \
                    OUTPUT_DIR  output/rpn/one-shot-split1/base \
                    MODEL.WEIGHTS weights/initial/rpn/one-shot-split1/model_final.pth \
                    DATASETS.TEST '("coco_2017_val_oneshot_s1",)'
        else
            python3 tools/train_net.py --eval-only \
                    --num-gpus 4  \
                    --config-file  configs/COCO-Detection/rpn_R_50_C4_1x.yaml \
                    OUTPUT_DIR  output/rpn/one-shot-split1 \
                    MODEL.WEIGHTS weights/initial/rpn/one-shot-split1/model_final.pth \
                    DATASETS.TEST '("coco_2017_val",)'
        fi
        ;;

        os2)
        if [[ "$2" == "base" ]]
        then
            python3 tools/train_net.py --eval-only \
                    --num-gpus 4  \
                    --config-file  configs/COCO-Detection/rpn_R_50_C4_1x.yaml \
                    OUTPUT_DIR  output/rpn/one-shot-split2/base \
                    MODEL.WEIGHTS  weights/initial/rpn/one-shot-split2/model_final.pth \
                    DATASETS.TEST '("coco_2017_val_oneshot_s2",)'
        else
            python3 tools/train_net.py --eval-only \
                    --num-gpus 4  \
                    --config-file  configs/COCO-Detection/rpn_R_50_C4_1x.yaml \
                    OUTPUT_DIR  output/rpn/one-shot-split2 \
                    MODEL.WEIGHTS weights/initial/rpn/one-shot-split2/model_final.pth \
                    DATASETS.TEST '("coco_2017_val",)'
        fi
        ;;

        os3)
        if [[ "$2" == "base" ]]
        then
            python3 tools/train_net.py --eval-only \
                    --num-gpus 4  \
                    --config-file  configs/COCO-Detection/rpn_R_50_C4_1x.yaml \
                    OUTPUT_DIR  output/rpn/one-shot-split3/base \
                    MODEL.WEIGHTS weights/initial/rpn/one-shot-split3/model_final.pth \
                    DATASETS.TEST '("coco_2017_val_oneshot_s3",)'
        else
            python3 tools/train_net.py --eval-only \
                    --num-gpus 4  \
                    --config-file  configs/COCO-Detection/rpn_R_50_C4_1x.yaml \
                    OUTPUT_DIR  output/rpn/one-shot-split3 \
                    MODEL.WEIGHTS weights/initial/rpn/one-shot-split3/model_final.pth \
                    DATASETS.TEST '("coco_2017_val",)'
        fi
        ;;

        os4)
        if [[ "$2" == "base" ]]
        then
            python3 tools/train_net.py --eval-only \
                    --num-gpus 4  \
                    --config-file  configs/COCO-Detection/rpn_R_50_C4_1x.yaml \
                    OUTPUT_DIR  output/rpn/one-shot-split4/base \
                    MODEL.WEIGHTS weights/initial/rpn/one-shot-split4/model_final.pth \
                    DATASETS.TEST '("coco_2017_val_oneshot_s4",)'
        else
            python3 tools/train_net.py --eval-only \
                    --num-gpus 4  \
                    --config-file  configs/COCO-Detection/rpn_R_50_C4_1x.yaml \
                    OUTPUT_DIR  output/rpn/one-shot-split4 \
                    MODEL.WEIGHTS  weights/initial/rpn/one-shot-split4/model_final.pth \
                    DATASETS.TEST '("coco_2017_val",)'
        fi
        ;;

        fs14)
        if [[ "$2" == "base" ]]
        then
            python3 tools/train_net.py --eval-only \
                    --num-gpus 4  \
                    --config-file  configs/COCO-Detection/rpn_R_50_C4_1x.yaml \
                    OUTPUT_DIR  output/rpn/few-shot-coco14/base \
                    MODEL.WEIGHTS weights/initial/rpn/few-shot-coco14/model_final.pth \
                    DATASETS.TEST '("fs_coco17_base_val",)'
        else
            python3 tools/train_net.py --eval-only \
                    --num-gpus 4  \
                    --config-file  configs/COCO-Detection/rpn_R_50_C4_1x.yaml \
                    OUTPUT_DIR  output/rpn/few-shot-coco14 \
                    MODEL.WEIGHTS weights/initial/rpn/few-shot-coco14/model_final.pth \
                    DATASETS.TEST '("fs_coco_test_all",)'
        fi
        ;;
        *)
        echo "no eval"
        ;;
esac

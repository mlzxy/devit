
case $1 in

        ovd)
        python3 tools/train_net.py \
                --num-gpus 4  \
                --config-file configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
                OUTPUT_DIR  output/mrcnn_rpn_ovd 
        ;;


        os1)
        python3 tools/train_net.py \
                --num-gpus 4  \
                --config-file  configs/RPN/mask_rcnn_R_50_C4_1x_oneshot_s1.yaml \
                OUTPUT_DIR  output/mrcnn_rpn_oneshot_s1 \
        ;;

        os2)
        python3 tools/train_net.py \
                --num-gpus 4  \
                --config-file configs/RPN/mask_rcnn_R_50_C4_1x_oneshot_s2.yaml \
                OUTPUT_DIR  output/mrcnn_rpn_oneshot_s2 \
        ;;

        os3)
        python3 tools/train_net.py \
                --num-gpus 4  \
                --config-file  configs/RPN/mask_rcnn_R_50_C4_1x_oneshot_s3.yaml \
                OUTPUT_DIR  output/mrcnn_rpn_oneshot_s3 \
        ;;

        os4)
        python3 tools/train_net.py \
                --num-gpus 4  \
                --config-file  configs/RPN/mask_rcnn_R_50_C4_1x_oneshot_s4.yaml \
                OUTPUT_DIR  output/mrcnn_rpn_oneshot_s4 \
        ;;

        fs14)
        python3 tools/train_net.py \
                --num-gpus 4  \
                --config-file configs/RPN/mask_rcnn_R_50_C4_1x_fewshot_14.yaml \
                OUTPUT_DIR  output/mrcnn_rpn_fewshot14 \
        ;;

        *)
        echo "no training"
        ;;
esac


 set -o xtrace

 case $1 in

         1)
         python3 tools/train_net.py \
                 --num-gpus 4  \
                 --config-file  configs/VOC_RPN/faster_rcnn_R_50_C4.few_shot_s1.yaml \
                 OUTPUT_DIR  output/pascal_voc_mrcnn_rpn_s1 \
         ;;

         2)
         python3 tools/train_net.py \
                 --num-gpus 4  \
                 --config-file  configs/VOC_RPN/faster_rcnn_R_50_C4.few_shot_s2.yaml \
                 OUTPUT_DIR  output/pascal_voc_mrcnn_rpn_s2 \
         ;;

         3)
         python3 tools/train_net.py \
                 --num-gpus 4  \
                 --config-file  configs/VOC_RPN/faster_rcnn_R_50_C4.few_shot_s3.yaml \
                 OUTPUT_DIR  output/pascal_voc_mrcnn_rpn_s3 \
         ;;

         *)
         echo "no training"
         ;;
 esac
~
python3 fix_label.py \
    -opt options/uni_lora_cls.yml \
    --weight_path "home/user01/aiotlab/nmduong/BoneTumor/src/exp_cls_256_cls_rm_confused_case_with9/UNI_lora_cls/_best.pt" \
    --labels_dir "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/RAW_DATA/labels" \
    --images_dir "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/RAW_DATA/images" \
    --old_labels_dir "./infer/smooth/Visualizations" \
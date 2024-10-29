python infer_smooth.py \
    -opt options/UNI_lora/uni_lora_cls_heavy.yml \
    --weight_path '/home/user01/aiotlab/nmduong/BoneTumor/src/exp_cls_256_cls_retrain_with9_case68/UNI_lora_cls/_best.pt' \
    --labels_dir '/home/user01/aiotlab/nmduong/BoneTumor/RAW_DATA/labels' \
    --images_dir '/home/user01/aiotlab/nmduong/BoneTumor/RAW_DATA/images' \
    --outdir './infer/smooth_uni_last/'
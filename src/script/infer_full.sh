python infer_full.py \
    -opt options/stacked_model/uni_lora_bbdm.yml \
    --phase1_path '/mnt/disk4/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/src/exp/exp_cls_256_cls_retrain_with9/UNI_lora_cls/_best.pt' \
    --phase2_path '/mnt/disk4/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/BBDM2/results/PredIm-1-ysubx-concat-with9/LBBDM-f8/checkpoint/top_model_epoch_15.pth' \
    --labels_dir '/mnt/disk4/nmduong/Vin-Uni-Bone-/labels' \
    --images_dir '/mnt/disk4/nmduong/Vin-Uni-Bone-/images' \
    --outdir './infer/smooth_stacked/'
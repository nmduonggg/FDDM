python infer_full.py \
    -opt options/stacked_model/uni_lora_discrete_bbdm.yml \
    --phase1_path '/home/user01/aiotlab/nmduong/BoneTumor/src/exp_cls_256_cls_retrain_with9/UNI_lora_cls/_best.pt' \
    --phase2_path '/home/user01/aiotlab/nmduong/BoneTumor/BBDM2/results/PredIm-1-ysubx-concat-with9-discrete/BrownianBridge-Pathology/checkpoint/top_model_epoch_22.pth' \
    --labels_dir '/home/user01/aiotlab/nmduong/BoneTumor/RAW_DATA/labels' \
    --images_dir '/home/user01/aiotlab/nmduong/BoneTumor/RAW_DATA/images' \
    --outdir './infer/smooth_stacked_discrete_ensemble_not68/'
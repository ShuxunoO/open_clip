# 动态蒙版训练

## CLIP-ViT-14 模型
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 -m training.main \
    --batch-size 50 \
    --precision amp \
    --workers 8 \
    --report-to tensorboard \
    --save-frequency 1 \
    --logs="/disk1/shuxun/NFTSearch/models/clip/openclip_finetuning/logs" \
    --dataset-type csv \
    --csv-separator="," \
    --train-data /disk1/shuxun/Dataset/NFT1000/_index/NFT1000_mini_img_component_caption_training.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1000 \
    --lr=3e-5 \
    --wd=0.1 \
    --epochs=32 \
    --model ViT-L-14 \
    --pretrained /disk1/shuxun/NFTSearch/models/clip/ViT-L-14.pt

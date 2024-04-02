# 动态蒙版训练

## CLIP-ViT-14 模型
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 -m training.main \
    --batch-size 50 \
    --precision amp \
    --workers 6 \
    --save-frequency 2 \
    --logs="/mnt/main/baiwm/Img_Retrieval/models" \
    --dataset-type csv \
    --csv-separator="," \
    --train-data /mnt/main/baiwm/Img_Retrieval/DataSet/NFT1000/_index/NFT1000_mini_img_component_caption_training.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1500 \
    --lr=3e-5 \
    --wd=0.1 \
    --epochs=32 \
    --model ViT-L-14 \
    --pretrained /mnt/main/baiwm/Img_Retrieval/DataSet/models/clip/ViT-L-14.pt


## Meta-CLIP-ViT-14 模型
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 -m training.main \
    --batch-size 50 \
    --precision amp \
    --workers 10 \
    --save-frequency 2 \
    --save-most-recent \
    --logs="/mnt/main/baiwm/Img_Retrieval/models" \
    --dataset-type csv \
    --csv-separator="," \
    --train-data /mnt/main/baiwm/Img_Retrieval/DataSet/NFT1000/_index/NFT1000_mini_img_component_caption_training.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1500 \
    --lr=3e-5 \
    --wd=0.1 \
    --epochs=32 \
    --model ViT-L-14 \
    --pretrained /mnt/main/baiwm/Img_Retrieval/DataSet/models/clip/l14_400m.pt
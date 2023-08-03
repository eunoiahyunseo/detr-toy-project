# python main.py \
#     --device cpu \
#     --dataset_file face \
#     --data_path ./workspace/custom/dataset/ \
#     --resume weights/detr-r50-e632da11.pth \
#     --output_dir ./workspace/custom/output/ \

python test.py \
    --device cpu \
    --data_path ./workspace/custom/dataset/ \
    --resume weights/detr-r50-e632da11.pth \

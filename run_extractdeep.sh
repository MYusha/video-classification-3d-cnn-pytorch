CUDA_VISIBLE_DEVICES=0 \
python main.py \
--input video_list.txt \
--video_root $VIDEO_ROOT \
--output data/jsons/$DATASET/resnet_3d_34.json \
--model data/models/resnet-34-kinetics.pth \
--mode feature 

# mode feature allow model to omit last fc layers
# use feature after global average pooling



VIDEO_ROOT='../ucfdata/ucf_small_jpg/'
MODEL_DIR='../3D-ResNets-PyTorch/models/resnext-101-kinetics.pth'
# MODEL_DIR='pretrained_files/epoch300_split1_train/save_300.pth'
OUT_DIR='video_features/'
LIST_DIR='../ucfdata/ucf_small_lists/list_only_train.txt'
# LIST_DIR='subset_labels/video_list_split1_train.txt'
# LIST_DIR='video_list'
if [ ! -d $OUT_DIR ]; then
	mkdir $OUT_DIR
fi
Feature_NAME='ucf11_features_train_xx.json'
test -f $OUT_DIR$Feature_NAME && rm -r $OUT_DIR$Feature_NAME
CUDA_VISIBLE_DEVICES=1 \
python extract_feature.py \
--input $LIST_DIR \
--video_root $VIDEO_ROOT \
--output $OUT_DIR$Feature_NAME \
--model $MODEL_DIR \
--n_classes 400 \
--mode feature \
--clip_vid mean \
--down_rate 1 \
--model_name resnext \
--model_depth 101 \
--resnet_shortcut B

# mode feature allow model to omit last fc layers
# use feature after global average pooling



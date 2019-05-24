VIDEO_ROOT='../hmdbdata/hmdb51_jpg/'
MODEL_DIR='pretrained_files/epoch300_split1_train/save_300.pth'
OUT_DIR='video_features/'
LIST_DIR='subset_labels/'
mkdir $OUT_DIR
Feature_NAME='split1_train_features.json'
rm -r $OUT_DIR$Feature_NAME
CUDA_VISIBLE_DEVICES=3,4 \
python main.py \
--input $LIST_DIR'video_list_split1_train.txt' \
--video_root $VIDEO_ROOT \
--output $OUT_DIR$Feature_NAME \
--model $MODEL_DIR \
--mode feature \
--model_name resnext \
--model_depth 101 \
--resnet_shortcut B

# mode feature allow model to omit last fc layers
# use feature after global average pooling



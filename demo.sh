python main.py --input video_list --video_root ./videos/hmdb51 --output output_d2.json --model ./pretrained_files/resnext-101-kinetics-hmdb51_split1.pth --mode score --model_name resnext --model_depth 101 --resnet_shortcut B
# python generate_result_video/generate_result_video.py ./output_d2.json ./videos ./videos_pred ./class_names_list 0

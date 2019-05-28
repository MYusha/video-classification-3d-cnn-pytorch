import os
import sys
import json
import numpy as np
import torch
import pdb
from torch.autograd import Variable
from PIL import Image
import time
from opts import parse_opts
from model import generate_model
from mean import get_mean

def main(video_root,output_root):
    start_time = time.time()
    for class_name in os.listdir(video_root):
        if 'Split' in class_name:
            continue
        print(class_name)
        class_path = os.path.join(video_root, class_name)
        if not os.path.isdir(class_path):
            continue
        dst_class_path = os.path.join(output_root, class_name)
        if not os.path.exists(dst_class_path):
            os.makedirs(dst_class_path)
        for jpg_folder in os.listdir(class_path):
            vid_matrix = []
            jpg_path = os.path.join(class_path,jpg_folder)
            if len(os.listdir(jpg_path))>0:
                for img in os.listdir(jpg_path):
                    if img.endswith('.jpg'):
                        with Image.open(os.path.join(jpg_path, img)) as tmp:
                            # tmp = tmp.convert('RGB')
                            tmp = np.asarray(tmp)
                        vid_matrix.append(tmp)
                vid_matrix = np.stack(vid_matrix, axis=0)
                dst_matrix = os.path.join(dst_class_path, jpg_folder + '.npy')
                np.save(dst_matrix, vid_matrix)

    exc_time = time.time() - start_time
    print("--- %s seconds ---" % exc_time)



if __name__ == "__main__":
    video_root = sys.argv[1]
    output_root = sys.argv[2]
    main(video_root,output_root)
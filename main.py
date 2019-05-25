'''
Takes in video_list as input, which consist of paths to jpg files of all testing video.
Returns result stored in json file(a list of dictionaries):
Element can be original clip features or mean feature of a video
'''
import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
import time

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video

if __name__=="__main__":
    start_time = time.time()
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    print('please check: number of classes is {}'.format(opt.n_classes))
    assert opt.clip_vid in ['ori','mean']
    downrate = opt.down_rate

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    outputs = []
    for cnt, input_file in enumerate(input_files):
        # if (cnt % 100) ==0:
        #     print('on the {} video on the list'.format(cnt))
        video_path = os.path.join(opt.video_root, input_file)
        if os.path.exists(video_path):
            print(video_path)
            # subprocess.call('mkdir tmp', shell=True)
            # subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(video_path),
            #                 shell=True)
            video_name = os.path.basename(input_file)
            result = classify_video(video_path, video_name, class_names, model, opt, downrate)
            if opt.clip_vid == 'mean':
                vid_feature = []
                for clip in result['clips']:
                    # clip is a dictionary with keys "segment" and "features"
                    vid_feature.append(clip['features'])
                mean_feature = np.mean(vid_feature, axis=0)
                result['clips'] = []
                result['mean_feature'] = mean_feature
                outputs.append(result)
            elif opt.clip_vid == 'ori':
                outputs.append(result)

            # subprocess.call('rm -rf tmp', shell=True)
        else:
            print('{} does not exist'.format(input_file))

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)

    print("--- %s seconds ---" % (time.time() - start_time))

import os
import sys
import json
import subprocess
import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import time

from opts import parse_opts
from model import generate_model
from mean import get_mean

from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding


def generate_vid_feature(vid_matrix, vid_name, model, opt, downrate):
    '''
    Modified from function classify_video()
    takes in video matrix F(frames) x H(height) x W(width) x C(channels)
    output vector representation of video
    '''
    assert vid_matrix.ndim==4 and downrate <= 1 # sanity check
    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(),
                                 Normalize(opt.mean, [1, 1, 1])])
    temporal_transform = LoopPadding(opt.sample_duration)
    data = Video(vid_matrix, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration, down_rate=downrate)
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)
    video_outputs = []
    video_segments = []
    for i, (inputs, segments) in enumerate(data_loader):
        inputs = Variable(inputs, volatile=True)
        # input is of shape n x 3 x sample_duration x 112 x 112
        # torch: set the input volatile=True if only doing inference not back-propagation
        outputs = model(inputs)
        # output is of format n(batch size) x d(dimension of feature)
        video_outputs.append(outputs.cpu().data)
        video_segments.append(segments)
        # segments is of shape batch_size x 2
    video_outputs = np.concatenate(video_outputs,axis=0)
    video_segments = np.concatenate(video_segments,axis=0)
    mean_feature = np.mean(video_outputs,axis=0) # shape of (d, )
    result = {
        'video_name': vid_name
    }
    if opt.clip_vid=='mean':
        result['feature'] = mean_feature
    return result


def main():
    start_time = time.time()
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    print('please check: number of classes is {}'.format(opt.n_classes))
    assert opt.clip_vid in ['ori', 'mean']
    downrate = opt.down_rate

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    # takes in input video (jpg) paths and transform to F x H x W x C matrix
    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])
    outputs = []
    for cnt, input_file in enumerate(input_files):
        vid_matrix = []
        video_path = os.path.join(opt.video_root, input_file)
        if os.path.exists(video_path):
            print(video_path)
        video_name = os.path.basename(input_file)
        for img in os.listdir(video_path):
            tmp = cv2.imread(img)
            vid_matrix.append(tmp)
        vid_matrix = np.stack(vid_matrix,axis=0)
        result = generate_vid_feature(vid_matrix, video_name, model, opt, downrate)
        outputs.append(result)
    exc_time = time.time() - start_time
    print("--- %s seconds ---" % exc_time)
    filename = opt.output+str(exc_time)
    with open(filename, 'w') as f:
        json.dump(outputs, f)


if __name__ == "__main__":
    main()
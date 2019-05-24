import os
import sys
import json
import pdb
import subprocess
import numpy as np

if __name__ == '__main__':
    result_json = sys.argv[1]
    feature_root_path = sys.argv[2]

    with open(os.path.join(feature_root_path,result_json), 'r') as f:
        data = json.load(f)
    n_vids = len(data)
    # pdb.set_trace()
    print('number of total video features included is', n_vids)
    for clip_data in data:
        # clip_data contains features from same video's clips
        vid = clip_data['video']
        n_clips = len(clip_data['clips'])
        vid_feature = []
        for clip in clip_data['clips']:
            # clip is a dictionary with keys "segment" and "features"
            vid_feature.append(clip['features'])
        vid_feature = np.asarray(vid_feature)
        mean_feature = np.mean(vid_feature,axis=0)
        # assert mean_feature.ndim == 1
        print mean_feature.shape


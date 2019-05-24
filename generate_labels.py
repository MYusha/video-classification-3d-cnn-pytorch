from __future__ import print_function
import os
import sys
import json
import pandas as pd
import pdb


def generate_label_from_split_txt(txt_path, subset, split_num, class_file):
    class_list = [line.rstrip('\n') for line in open(class_file)]
    f = open('subset_labels/labels_'+str(subset)+str(split_num)+".txt", "w+")
    subset_to_num = {'test':2, 'train':1}
    total_vids = 0
    total_files = 0
    for filename in os.listdir(txt_path):
        if 'split{}'.format(split_num) not in filename:
            continue
        total_files += 1
        class_name = filename.split('_test')[0]
        data = pd.read_csv(os.path.join(txt_path,filename), delimiter=' ', header=None)
        for i in range(data.shape[0]):
            vid = data.ix[i, :]
            if vid[1] == subset_to_num[subset]:
                f.write(vid[0]+' '+str(class_list.index(class_name))+'\n')
                total_vids += 1
    f.close()
    print('summary of split {} of the {} set'.format(split_num,subset))
    print('totally {} videos'.format(total_vids))
    print('totally {} annotation files read'.format(total_files))
        # pdb.set_trace()

def generate_video_list_from_labels(label_file,subset,split_index, class_file):
    class_list = [line.rstrip('\n') for line in open(class_file)]
    f = open('subset_labels/video_list_split' + str(split_index) + '_'+str(subset)+".txt", "w+")
    with open(label_file) as fp:
        for cnt, line in enumerate(fp):
            # pdb.set_trace()
            img_file = os.path.splitext(line.split(' ')[0])[0]
            f.write(os.path.join(class_list[int(line.split(' ')[1])],img_file) + '\n')
    f.close()



if __name__ == '__main__':
    txt_dir_path = sys.argv[1]
    subset = sys.argv[2] #train or test
    class_file = sys.argv[3]
    assert subset in ['test','train']
    for split_index in range(1, 4):
        # generate_label_from_split_txt(txt_dir_path, subset, split_index, class_file)
        label_dir = 'subset_labels/labels_'+str(subset)+str(split_index)+".txt"
        generate_video_list_from_labels(label_dir,subset,split_index,class_file)


from __future__ import print_function, division
import os
import sys
import subprocess

def main(dir_path, dst_dir_path):
    for video_file in os.listdir(dir_path):
        if '.mp4' not in video_file:
            continue
        name, ext = os.path.splitext(video_file)
        dst_directory_path = os.path.join(dst_dir_path, name)
        if not os.path.exists(dst_directory_path):
            os.makedirs(dst_directory_path)
        video_file_path = os.path.join(dir_path, video_file)
        cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')

if __name__ == "__main__":
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]
    main(dir_path,dst_dir_path)



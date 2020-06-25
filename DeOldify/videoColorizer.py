import os
from DeOldify.deoldify.visualize import *

def get_colorizer():
    return get_video_colorizer()

def colorize_video(filename, colorizer):
    file_name_ext = filename
    result_path = None
    result_path = colorizer.colorize_from_file_name(file_name_ext)
    return result_path

if __name__ == '__main__':
    colorize_video(None)

from SepConv.run import improve_fps
from DeOldify.videoColorizer import colorize_video
import argparse
import torch
import os
import warnings
# Ignore deprecation warning thrown by pytorch
warnings.filterwarnings("ignore", category=UserWarning) 
torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="Sepconv in pytorch")
    parser.add_argument("--colorize", action="store_true")
    args = parser.parse_args()
    print("Is CUDA available? ", torch.cuda.is_available())

    if args.colorize:
        colorized_path = colorize_video()
        sepconv_path = colorized_path
    else:
        sepconv_path = os.path.join("video", "source", "input.mp4")
    print(colorized_path)
    result_file = improve_fps(1, sepconv_path)
    print("Improved video stored in ", result_file)

if __name__ == '__main__':
    main()
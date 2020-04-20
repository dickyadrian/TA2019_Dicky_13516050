from SepConv.forward import improve_fps
from SepConv.train import train
import argparse
import torch
import os
import warnings
# Ignore deprecation warning thrown by pytorch
warnings.filterwarnings("ignore", category=UserWarning) 

def main():
    parser = argparse.ArgumentParser(description="Sepconv in pytorch")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.train:
        train('db', 51, 'output_sepconv_pytorch', 10, 1, os.path.join('Interpolation_testset', 'input'), os.path.join('Interpolation_testset', 'gt'), load_model="./converted2.pth")
    else:
        result_file = improve_fps(1, 'test.mp4', 'result', './converted2.pth')
        print("Improved video stored in ", result_file)

if __name__ == '__main__':
    main()
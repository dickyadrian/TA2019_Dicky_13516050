import cv2
import numpy as np
import torch
import os
from .model import SepConvNet
from tqdm import tqdm
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

import sys

def cv_to_pil(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    else:
        raise Exception("Only CUDA supported device can run this file")
    return Variable(x)

def improve_fps (times, input_file, out_dir, model_path):
    result = os.path.join(out_dir, 'result.mp4')

    print('Reading input file...')
    cap = cv2.VideoCapture(input_file)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    old_fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    duration = frame_count / old_fps

    if times == 1:
        fps = ((2 * frame_count) - 1) / duration
    elif times == 2:
        fps = ((4 * frame_count) - 3) / duration
    elif times == 3:
        fps = ((8 * frame_count) - 7) / duration
    else:
        print("Times is not supported, proceeding without interpolation")
        fps = old_fps

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(result, fourcc, fps, (width, height))
    
    # Read first 2 frames
    succ1, frame1 = cap.read()
    succ2, frame2 = cap.read()
    if not(succ1 and succ2):
        raise Exception("Failed reading frames, video may contain less than 2 frames")
    
    print("Loading model ...")
    checkpoint = torch.load(model_path)
    kernel_size = checkpoint['kernel_size']
    model = SepConvNet(kernel_size=kernel_size)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.epoch = checkpoint['epoch']
    model.cuda()
    transform = transforms.Compose([transforms.ToTensor()])

    print("Forward Start...")
    frames = [frame1, frame2]
    for i in tqdm(range(int(frame_count-1))):
        for j in range(times):
            for k in range(len(frames) - 1):
                firstImage = to_variable(transform(cv_to_pil(frames[k])).unsqueeze(0))
                secondImage = to_variable(transform(cv_to_pil(frames[k+1])).unsqueeze(0))
                frame_out = model(firstImage, secondImage)
                frame_out = frame_out.cpu().squeeze()
                im_out = (frame_out.clamp(0.0, 1.0).detach().numpy().transpose(1,2,0)[:, :, ::-1] * 255.0).astype(np.uint8)
                frames.insert(k+1, im_out)
        
        if (i == 0):
            for frame in frames:
                video.write(frame)
        else:
            for j in range(1, len(frames)):
                video.write(frames[j])
        
        frame1 = frame2
        succ2, frame2 = cap.read()
        if not succ2:
            break
        frames = [frame1, frame2]
    
    video.release()
    cv2.destroyAllWindows()
    return result
import time
import os
import sys
from torch.autograd import Variable
import math
import torch

import random
import numpy as np
import numpy
import networks
from my_args import  args
import cv2
from tqdm import tqdm

torch.backends.cudnn.benchmark = True # to speed up the


DO_MiddleBurryOther = True
MB_Other_DATA = "./VideoData"
MB_Other_RESULT = "./Interpolated"
MB_Other_GT = "./MiddleBurySet/other-gt-interp/"
VIDEO_FILE = './video.mp4'
OUT_NAME = './video_hasil.mp4'
if not os.path.exists(MB_Other_RESULT):
    os.mkdir(MB_Other_RESULT)



model = networks.__dict__[args.netName](channel=args.channels,
                            filter_size = args.filter_size ,
                            timestep=args.time_step,
                            training=False)

if args.use_cuda:
    model = model.cuda()

args.SAVED_MODEL = './model_weights/best.pth'
if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
        # model.load_state_dict(torch.load(args.SAVED_MODEL))

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We don't load any trained weights **************************")
    print("*****************************************************************")

model = model.eval() # deploy mode


use_cuda=args.use_cuda
save_which=args.save_which
dtype = args.dtype
unique_id =str(random.randint(0, 100000))
print("The unique id for current testing is: " + str(unique_id))

if DO_MiddleBurryOther:
    end = time.time()
    vidcap = cv2.VideoCapture(VIDEO_FILE)
    fps_old = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps_old
    success1, frame1 = vidcap.read()
    success2, frame2 = vidcap.read()
    if (not success1 or not success2):
        print('Video is less than 2 frames, exiting...')
        sys.exit()

    fps_new = ((2*frame_count)-1)/duration
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(OUT_NAME, fourcc, fps_new, (frame1.shape[1], frame2.shape[0]))
    for image in file_paths:
        video.write(cv2.imread(image))
    for i in tqdm(range(frame_count-1)):

        X0 =  torch.from_numpy( np.transpose(frame1 , (2,0,1)).astype("float32")/ 255.0).type(dtype)
        X1 =  torch.from_numpy( np.transpose(frame2 , (2,0,1)).astype("float32")/ 255.0).type(dtype)

        y_ = torch.FloatTensor()

        assert (X0.size(1) == X1.size(1))
        assert (X0.size(2) == X1.size(2))

        intWidth = X0.size(2)
        intHeight = X0.size(1)
        channel = X0.size(0)
        if not channel == 3:
            continue

        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft =int(( intWidth_pad - intWidth)/2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 32
            intPaddingRight= 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

        torch.set_grad_enabled(False)
        X0 = Variable(torch.unsqueeze(X0,0))
        X1 = Variable(torch.unsqueeze(X1,0))
        X0 = pader(X0)
        X1 = pader(X1)

        if use_cuda:
            X0 = X0.cuda()
            X1 = X1.cuda()
        y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
        y_ = y_s[save_which]

        if use_cuda:
            X0 = X0.data.cpu().numpy()
            y_ = y_.data.cpu().numpy()
            offset = [offset_i.data.cpu().numpy() for offset_i in offset]
            filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
            X1 = X1.data.cpu().numpy()
        else:
            X0 = X0.data.numpy()
            y_ = y_.data.numpy()
            offset = [offset_i.data.numpy() for offset_i in offset]
            filter = [filter_i.data.numpy() for filter_i in filter]
            X1 = X1.data.numpy()

        X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
        y_ = np.transpose(255.0 * y_.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
        offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
        filter = [np.transpose(
            filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
            (1, 2, 0)) for filter_i in filter]  if filter is not None else None
        X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

        inter_result = np.round(y_).astype(numpy.uint8)
        if i == 0:
            video.write(frame1)
            video.write(inter_result)
            video.write(frame2)
        else:
            video.write(inter_result)
            video.write(frame2)
        
        frame1 = frame2
        success2, frame2 = vidcap.read()

        if not success2:
            break
    cv2.destroyAllWindows()
    video.release()
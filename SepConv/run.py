#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import random
import shutil
import sys
import tempfile
import cv2
from tqdm import tqdm
from .sepconv.sepconv import FunctionSepconv

# end

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		def Basic(intInput, intOutput):
			return torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False)
			)
		# end

		def Upsample(intInput, intOutput):
			return torch.nn.Sequential(
				torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
				torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False)
			)
		# end

		def Subnet():
			return torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
				torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
			)
		# end

		self.moduleConv1 = Basic(6, 32)
		self.moduleConv2 = Basic(32, 64)
		self.moduleConv3 = Basic(64, 128)
		self.moduleConv4 = Basic(128, 256)
		self.moduleConv5 = Basic(256, 512)

		self.moduleDeconv5 = Basic(512, 512)
		self.moduleDeconv4 = Basic(512, 256)
		self.moduleDeconv3 = Basic(256, 128)
		self.moduleDeconv2 = Basic(128, 64)

		self.moduleUpsample5 = Upsample(512, 512)
		self.moduleUpsample4 = Upsample(256, 256)
		self.moduleUpsample3 = Upsample(128, 128)
		self.moduleUpsample2 = Upsample(64, 64)

		self.moduleVertical1 = Subnet()
		self.moduleVertical2 = Subnet()
		self.moduleHorizontal1 = Subnet()
		self.moduleHorizontal2 = Subnet()

		if torch.cuda.is_available():
			self.load_state_dict(torch.load(__file__.replace('run.py', 'network-lf' + '.pytorch')))
		else:
			self.load_state_dict(torch.load(__file__.replace('run.py', 'network-lf' + '.pytorch'), map_location='cpu'))
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorConv1 = self.moduleConv1(torch.cat([ tensorFirst, tensorSecond ], 1))
		tensorConv2 = self.moduleConv2(torch.nn.functional.avg_pool2d(input=tensorConv1, kernel_size=2, stride=2, count_include_pad=False))
		tensorConv3 = self.moduleConv3(torch.nn.functional.avg_pool2d(input=tensorConv2, kernel_size=2, stride=2, count_include_pad=False))
		tensorConv4 = self.moduleConv4(torch.nn.functional.avg_pool2d(input=tensorConv3, kernel_size=2, stride=2, count_include_pad=False))
		tensorConv5 = self.moduleConv5(torch.nn.functional.avg_pool2d(input=tensorConv4, kernel_size=2, stride=2, count_include_pad=False))

		tensorDeconv5 = self.moduleUpsample5(self.moduleDeconv5(torch.nn.functional.avg_pool2d(input=tensorConv5, kernel_size=2, stride=2, count_include_pad=False)))
		tensorDeconv4 = self.moduleUpsample4(self.moduleDeconv4(tensorDeconv5 + tensorConv5))
		tensorDeconv3 = self.moduleUpsample3(self.moduleDeconv3(tensorDeconv4 + tensorConv4))
		tensorDeconv2 = self.moduleUpsample2(self.moduleDeconv2(tensorDeconv3 + tensorConv3))

		tensorCombine = tensorDeconv2 + tensorConv2
		del tensorConv1, tensorConv2, tensorConv3, tensorConv4, tensorConv5, tensorDeconv5, tensorDeconv4, tensorDeconv3, tensorDeconv2

		tensorFirst = torch.nn.functional.pad(input=tensorFirst, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ], mode='replicate')
		tensorSecond = torch.nn.functional.pad(input=tensorSecond, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ], mode='replicate')

		tensorDot1 = FunctionSepconv(tensorInput=tensorFirst, tensorVertical=self.moduleVertical1(tensorCombine), tensorHorizontal=self.moduleHorizontal1(tensorCombine))
		del tensorFirst
		tensorDot2 = FunctionSepconv(tensorInput=tensorSecond, tensorVertical=self.moduleVertical2(tensorCombine), tensorHorizontal=self.moduleHorizontal2(tensorCombine))
		del tensorSecond

		return tensorDot1 + tensorDot2
	# end
# end

moduleNetwork = None

##########################################################

def estimate(tensorFirst, tensorSecond):
	global moduleNetwork

	if moduleNetwork is None:
		if torch.cuda.is_available():
			moduleNetwork = Network().cuda().eval()
		else:
			moduleNetwork = Network().eval()
	# end

	intWidth = tensorFirst.shape[2]
	intHeight = tensorFirst.shape[1]

	tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
	tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

	intPaddingLeft, intPaddingTop, intPaddingBottom, intPaddingRight = 0, 0, 0, 0

	# end

	intPreprocessedWidth = intPaddingLeft + intWidth + intPaddingRight
	intPreprocessedHeight = intPaddingTop + intHeight + intPaddingBottom

	if intPreprocessedWidth != ((intPreprocessedWidth >> 7) << 7):
		intPreprocessedWidth = (((intPreprocessedWidth >> 7) + 1) << 7) # more than necessary
	# end
	
	if intPreprocessedHeight != ((intPreprocessedHeight >> 7) << 7):
		intPreprocessedHeight = (((intPreprocessedHeight >> 7) + 1) << 7) # more than necessary
	# end

	intPaddingRight = intPreprocessedWidth - intWidth - intPaddingLeft
	intPaddingBottom = intPreprocessedHeight - intHeight - intPaddingTop

	tensorPreprocessedFirst = torch.nn.functional.pad(input=tensorPreprocessedFirst, pad=[ intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom ], mode='replicate')
	tensorPreprocessedSecond = torch.nn.functional.pad(input=tensorPreprocessedSecond, pad=[ intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom ], mode='replicate')

	return torch.nn.functional.pad(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), pad=[ 0 - intPaddingLeft, 0 - intPaddingRight, 0 - intPaddingTop, 0 - intPaddingBottom ], mode='replicate')[0, :, :, :].cpu()
# end

##########################################################

def improve_fps(times, video_path):
	result = os.path.join('video', 'result', 'result.mp4')

	cap = cv2.VideoCapture(video_path)
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
		fps = old_fps
	fps = round(fps)
	fourcc = cv2.VideoWriter_fourcc(*'H264')
	video = cv2.VideoWriter(result, fourcc, fps, (width, height))

	succ1, frame1 = cap.read()
	succ2, frame2 = cap.read()

	if not(succ1 and succ2):
		return 'FAILED'

	frames = [frame1, frame2]

	with torch.no_grad():
		for i in tqdm(range(int(frame_count-1))):
			for j in range(times):
				for k in range(len(frames) - 1):
					tensorFirst = torch.FloatTensor(frames[k][:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
					tensorSecond = torch.FloatTensor(frames[k+1][:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

					tensorOutput = estimate(tensorFirst, tensorSecond)
					im_out = (tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)
					frames.insert(k+1, im_out)

			if i == 0:
				for frame in frames:
					video.write(frame)
			else:
				for l in range(1, len(frames)):
					video.write(frames[l])

			frame1 = frame2
			success, frame2 = cap.read()
			
			if not success:
				break
			frames = [frame1, frame2]
	video.release()
	cv2.destroyAllWindows()
	return result

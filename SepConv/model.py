import torch
import torch.optim as optim
from torch.autograd import Variable
import math
from .sepconv import *
import sys
from torch.nn import functional as F
import time


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        def Basic(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Upsample(channel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Subnet(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = Upsample(512)

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = Upsample(256)

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = Upsample(128)

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = Upsample(64)

        self.moduleVertical1 = Subnet(self.kernel_size)
        self.moduleVertical2 = Subnet(self.kernel_size)
        self.moduleHorizontal1 = Subnet(self.kernel_size)
        self.moduleHorizontal2 = Subnet(self.kernel_size)

    def forward(self, rfield0, rfield2):
        tensorJoin = torch.cat([rfield0, rfield2], 1)
        print("tensorJoin shape:", tensorJoin.shape)

        print("Start Conv1")
        start = time.time()
        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)
        print("Ended in ", time.time() - start, "shape: ", tensorPool1.shape)

        print("Start Conv2")
        start = time.time()
        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)
        print("Ended in ", time.time() - start, "shape: ", tensorPool2.shape)

        print("Start Conv3")
        start = time.time()
        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)
        print("Ended in ", time.time() - start, "shape: ", tensorPool3.shape)

        print("Start Conv4")
        start = time.time()
        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)
        print("Ended in ", time.time() - start, "shape: ", tensorPool4.shape)

        print("Start Conv5")
        start = time.time()
        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)
        print("Ended in ", time.time() - start, "shape: ", tensorPool5.shape)

        print("Start deconv5")
        start = time.time()
        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)
        print("Ended in ", time.time() - start, "shape: ", tensorUpsample5.shape)

        tensorCombine = tensorUpsample5 + tensorConv5

        print("Start deconv4")
        start = time.time()
        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)
        print("Ended in ", time.time() - start, "shape: ", tensorUpsample4.shape)

        tensorCombine = tensorUpsample4 + tensorConv4

        print("Start deconv3")
        start = time.time()
        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        print("Ended in ", time.time() - start, "shape: ", tensorUpsample3.shape)

        tensorCombine = tensorUpsample3 + tensorConv3

        print("Start deconv2")
        start = time.time()
        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        print("Ended in ", time.time() - start, "shape: ", tensorUpsample2.shape)        

        tensorCombine = tensorUpsample2 + tensorConv2

        print("Start vertical1")
        start = time.time()
        Vertical1 = self.moduleVertical1(tensorCombine)
        print("Ended in ", time.time() - start, "shape: ", Vertical1.shape)

        print("Start vertical2")
        start = time.time()
        Vertical2 = self.moduleVertical2(tensorCombine)
        print("Ended in ", time.time() - start, "shape: ", Vertical2.shape)

        print("Start horizontal1")
        start = time.time()
        Horizontal1 = self.moduleHorizontal1(tensorCombine)
        print("Ended in ", time.time() - start, "shape: ", Horizontal1.shape)

        print("Start horizontal2")
        start = time.time()
        Horizontal2 = self.moduleHorizontal2(tensorCombine)
        print("Ended in ", time.time() - start, "shape: ", Horizontal2.shape)

        return Vertical1, Horizontal1, Vertical2, Horizontal2


class SepConvNet(torch.nn.Module):
    def __init__(self, kernel_size):
        super(SepConvNet, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_pad = int(math.floor(kernel_size / 2.0))

        self.epoch = Variable(torch.tensor(0, requires_grad=False))
        self.get_kernel = KernelEstimation(self.kernel_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = torch.nn.L1Loss(reduction='sum')

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame0 = F.pad(frame0, (0, 0, 0, pad_h))
            frame2 = F.pad(frame2, (0, 0, 0, pad_h))
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame0 = F.pad(frame0, (0, pad_w, 0, 0))
            frame2 = F.pad(frame2, (0, pad_w, 0, 0))
            w_padded = True

        Vertical1, Horizontal1, Vertical2, Horizontal2 = self.get_kernel(frame0, frame2)
        
        start = time.time()
        tensorDot1 = FunctionSepconv()(self.modulePad(frame0), Vertical1, Horizontal1)
        print(tensorDot1.shape, "Tensordot1 shape, finish in: ", time.time() - start)
        start = time.time()
        tensorDot2 = FunctionSepconv()(self.modulePad(frame2), Vertical2, Horizontal2)
        print(tensorDot2.shape, "Tensordot2 shape, finish in: ", time.time() - start)

        frame1 = tensorDot1 + tensorDot2

        if h_padded:
            frame1 = frame1[:, :, 0:h0, :]
        if w_padded:
            frame1 = frame1[:, :, :, 0:w0]

        return frame1

    def train_model(self, frame0, frame2, frame1):
        self.optimizer.zero_grad()
        output1 = self.forward(frame0, frame2)
        output2 = self.forward(frame0, frame1)
        output3 = self.forward(frame1, frame2)
        output4 = self.forward(output2, output3)
        loss1 = self.criterion(output1, frame1)
        loss2 = self.criterion(output4, frame1)
        total_loss = sum([loss1, loss2])
        total_loss.backward()
        self.optimizer.step()
        return total_loss

    def increase_epoch(self):
        self.epoch += 1

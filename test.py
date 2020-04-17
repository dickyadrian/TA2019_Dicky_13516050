from SepConv.model import SepConvNet
import torch
from torch.autograd import Variable

if __name__ == "__main__":
    model = SepConvNet(kernel_size=51)
    model.epoch = Variable(torch.tensor(0, requires_grad=False))
    model.get_kernel.load_state_dict(torch.load('./network-lf.pytorch'))
    torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), 'kernel_size': 51}, './converted2.pth')
from SepConv.forward import *
import torch

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
 
if __name__ == '__main__':
  # if torch.cuda.is_available():
	# 		moduleNetwork = Network('./models/network-lf.pytorch').cuda().eval()
	# 	else:
	# 		moduleNetwork = Network('./models/network-lf.pytorch').eval()
  print(__file__)
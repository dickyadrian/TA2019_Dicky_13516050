from SepConv.forward import *
import torch
import time

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
 
if __name__ == '__main__':
  start = time.time()
  # Check if CUDA is enabled
  if torch.cuda.is_available():
    moduleNetwork = Network('./models/network-lf.pytorch').cuda().eval()
  else:
    raise NotImplementedError() #Inference without CUDA is not implemented
    moduleNetwork = Network('./models/network-lf.pytorch').eval()
  print("Model loaded in: ", (time.time()-start))
  
  start = time.time()
  out = improve_fps(2, './test.mp4', './', moduleNetwork)
  print("Inference done in: ", (time.time()-start))

  print("Result stored in ", out)
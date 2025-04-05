import torch
import torchvision
import dlib
print("Torch Version:", torch.__version__)
print("TorchVision Version:", torchvision.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device:", torch.cuda.get_device_name(0))
print("Settings")
print(dlib.DLIB_USE_CUDA)

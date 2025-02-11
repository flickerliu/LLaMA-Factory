import torch

print(f'Torch Version: {torch.__version__}')
print(f'Is CUDA Enbaled for Torch:{torch.cuda.is_available()}') 

print(f'Device Count:{torch._C._cuda_getDeviceCount()}')
print(f'Current Device:{torch.cuda.current_device()}')
print(f'Device Name:{torch.cuda.get_device_name(0)}')
torch.__version__
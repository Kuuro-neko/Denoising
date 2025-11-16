import torch
print("CUDA Information")
print("Version : ", torch.version.cuda)
print("Is built : ", torch.backends.cuda.is_built())
print("Is available : ", torch.cuda.is_available())
print("Device : ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
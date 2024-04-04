import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
    
    # Check the number of GPUs available
    num_gpus = torch.cuda.device_count()
    # List the names of the GPUs
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

else:
    device = torch.device("cpu")
    print("GPU is not available")
    print("Using CPU instead")


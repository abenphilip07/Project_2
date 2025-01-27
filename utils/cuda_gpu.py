import torch

def check_cuda_gpu():
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get current device
        current_device = torch.cuda.current_device()
        # Get device name
        device_name = torch.cuda.get_device_name(current_device)
        # Get device count
        device_count = torch.cuda.device_count()
        
        print(f"Current CUDA Device: {current_device}")
        print(f"Device Name: {device_name}")
        print(f"Number of CUDA Devices: {device_count}")
        print(f"Device Properties: {torch.cuda.get_device_properties(current_device)}")
    else:
        print("No CUDA GPU available. Using CPU.")

if __name__ == "__main__":
    check_cuda_gpu()
import os
import sys
import argparse
import safetensors
import torch

def main():
    parser = argparse.ArgumentParser(description="Extract keys and precision from model files.")
    parser.add_argument("folder", help="Folder path containing model files")
    args = parser.parse_args()
    
    folder_path = args.folder
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    # Get all model files
    model_files = []
    for file in os.listdir(folder_path):
        if file.endswith((".ckpt", ".safetensors")):
            model_files.append(os.path.join(folder_path, file))
    
    if not model_files:
        print(f"No .ckpt or .safetensors files found in {folder_path}")
        sys.exit(0)
    
    print(f"Found {len(model_files)} model file(s) to process")
    
    # Process each file
    for file_path in model_files:
        file_name = os.path.basename(file_path)
        log_path = os.path.join(folder_path, f"{file_name}.keys.log")
        
        print(f"Processing {file_name}...")
        
        try:
            # Extract keys and precision based on file type
            if file_path.endswith(".safetensors"):
                with open(log_path, "w") as log_file:
                    with safetensors.safe_open(file_path, "pt") as f:
                        keys = list(f.keys())
                        for key in keys:
                            tensor = f.get_tensor(key)
                            precision = str(tensor.dtype)
                            shape = str(tensor.shape)
                            log_file.write(f"{key}: {precision} {shape}\n")
                print(f"  Extracted {len(keys)} keys with precision to {os.path.basename(log_path)}")
                            
            elif file_path.endswith(".ckpt"):
                checkpoint = torch.load(file_path, map_location="cpu")
                with open(log_path, "w") as log_file:
                    if isinstance(checkpoint, dict):
                        if "state_dict" in checkpoint:
                            state_dict = checkpoint["state_dict"]
                        else:
                            state_dict = checkpoint
                        
                        for key, tensor in state_dict.items():
                            if hasattr(tensor, 'dtype'):
                                precision = str(tensor.dtype)
                                shape = str(tensor.shape)
                                log_file.write(f"{key}: {precision} {shape}\n")
                            else:
                                log_file.write(f"{key}: unknown\n")
                        
                        print(f"  Extracted {len(state_dict)} keys with precision to {os.path.basename(log_path)}")
            
        except Exception as e:
            print(f"  Error processing {file_name}: {e}")

if __name__ == "__main__":
    main()

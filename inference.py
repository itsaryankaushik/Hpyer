import os
import torch
import torch.nn.parallel.data_parallel
import torch.serialization
import argparse
import numpy as np
import h5py
from PIL import Image
from Hyper import HyperFusion
from preprocess import preprocess_single_file, postprocess_data

# Add DataParallel to safe globals
torch.serialization.add_safe_globals([torch.nn.parallel.data_parallel.DataParallel])

def load_model(model_path, device='cuda'):
    """
    Load the HyperFusion model from checkpoint
    
    Args:
        model_path (str): Path to the model checkpoint
        device (str): Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        HyperFusion: Loaded model
    """
    # Initialize model with default parameters
    model = HyperFusion(inch=3, dim=64, upscale=4)
    
    # Load checkpoint with weights_only=False for backward compatibility
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Warning: Initial load failed with error: {e}")
        print("Attempting to load with weights_only=True...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

def save_output(data, output_path, data_type='image'):
    """
    Save processed data in appropriate format
    
    Args:
        data: Processed data (PIL Image or numpy array)
        output_path (str): Path to save the output
        data_type (str): Type of output data ('image', 'h5', 'matrix')
    """
    if data_type == 'image':
        data.save(output_path)
    elif data_type == 'h5':
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('data', data=data)
    else:  # matrix
        if output_path.endswith('.npy'):
            np.save(output_path, data)
        else:  # .mat
            from scipy.io import savemat
            savemat(output_path, {'data': data})

def process_file(model, input_path, output_path, device='cuda', data_type='image'):
    """
    Process a single file through the model
    
    Args:
        model (HyperFusion): Loaded model
        input_path (str): Path to input file
        output_path (str): Path to save output
        device (str): Device to run inference on
        data_type (str): Type of data file ('image', 'h5', 'matrix')
    """
    # Preprocess file
    input_tensor = preprocess_single_file(input_path, data_type=data_type)
    input_tensor = input_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Postprocess and save
    output_data = postprocess_data(output, data_type=data_type)
    save_output(output_data, output_path, data_type=data_type)
    print(f"Processed file saved to {output_path}")

def process_directory(model, input_dir, output_dir, device='cuda', data_type='image'):
    """
    Process all files in a directory
    
    Args:
        model (HyperFusion): Loaded model
        input_dir (str): Directory containing input files
        output_dir (str): Directory to save output files
        device (str): Device to run inference on
        data_type (str): Type of data files ('image', 'h5', 'matrix')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file extensions based on data type
    if data_type == 'image':
        extensions = ('.png', '.jpg', '.jpeg')
    elif data_type == 'h5':
        extensions = ('.h5',)
    else:  # matrix
        extensions = ('.npy', '.mat')
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(extensions):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"enhanced_{filename}"
            if data_type == 'h5':
                output_filename = output_filename.replace('.h5', '.h5')
            elif data_type == 'matrix':
                if filename.endswith('.npy'):
                    output_filename = output_filename.replace('.npy', '.npy')
                else:
                    output_filename = output_filename.replace('.mat', '.mat')
            output_path = os.path.join(output_dir, output_filename)
            process_file(model, input_path, output_path, device, data_type)

def main():
    parser = argparse.ArgumentParser(description='HyperFusion Data Enhancement Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output file or directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--data_type', type=str, default='image', 
                      choices=['image', 'h5', 'matrix'],
                      help='Type of data files (image/h5/matrix)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Process input
    if os.path.isdir(args.input):
        process_directory(model, args.input, args.output, device, args.data_type)
    else:
        process_file(model, args.input, args.output, device, args.data_type)

if __name__ == '__main__':
    main() 
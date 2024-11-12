import numpy as np

def print_npz_contents(file_path):
    """
    Read and print the contents of an NPZ file, showing at least two samples for each element.
    Calculate mean and standard deviation of position and orientation errors for converged samples.
    Show the top 5 lowest errors for both position and orientation.
    
    Args:
    file_path (str): Path to the NPZ file
    """
    # Load the NPZ file
    data = np.load(file_path, allow_pickle=True)
    
    print("Contents of the NPZ file:")
    print("-------------------------")
    
    position_errors = None
    orientation_errors = None
    convergence_flags = None

    for key in data.files:
        print(f"\nKey: {key}")
        value = data[key]
        
        if key == "position_errors":
            position_errors = value
        elif key == "orientation_errors":
            orientation_errors = value
        elif key == "convergence_flags":
            convergence_flags = value
        
        if isinstance(value, np.ndarray):
            print(f"Type: numpy.ndarray")
            print(f"Shape: {value.shape}")
            print(f"Data type: {value.dtype}")
            
            # Print samples of the data
            if value.dtype == object:
                print("Samples:")
                for i in range(min(2, len(value))):
                    print(f"  Element {i}:")
                    if isinstance(value[i], np.ndarray):
                        print(f"    Shape: {value[i].shape}")
                        if value[i].ndim == 1:
                            print(f"    Data: {value[i][:5]} ...")
                        else:
                            print(f"    First few rows:")
                            print(value[i][:5])
                    else:
                        print(f"    Data: {value[i]}")
            else:
                print("Samples (first few elements):")
                if value.ndim == 1:
                    print(value[:5])
                else:
                    print(value[:2])
        else:
            print(f"Type: {type(value)}")
            print("Value:")
            print(value)
        
        print("-------------------------")

    # Calculate statistics for converged samples
    if position_errors is not None and orientation_errors is not None and convergence_flags is not None:
        converged_mask = convergence_flags == True
        converged_position_errors = position_errors[converged_mask]
        converged_orientation_errors = orientation_errors[converged_mask]

        total_samples = len(convergence_flags)
        converged_samples = np.sum(converged_mask)
        convergence_percentage = (converged_samples / total_samples) * 100

        print("\nError Statistics (for converged samples only):")
        print(f"Convergence: {converged_samples} out of {total_samples} ({convergence_percentage:.2f}%)")
        
        print("\nPosition Errors:")
        print(f"Mean: {np.mean(converged_position_errors):.6f}")
        print(f"Standard Deviation: {np.std(converged_position_errors):.6f}")
        
        print("\nOrientation Errors:")
        print(f"Mean: {np.mean(converged_orientation_errors):.6f}")
        print(f"Standard Deviation: {np.std(converged_orientation_errors):.6f}")

        # Find and display the lowest errors
        print("\nTop 5 Lowest Position Errors:")
        lowest_position_indices = np.argsort(converged_position_errors)[:5]
        for i, idx in enumerate(lowest_position_indices):
            print(f"  {i+1}. Sample {idx}: {converged_position_errors[idx]:.6f}")

        print("\nTop 5 Lowest Orientation Errors:")
        lowest_orientation_indices = np.argsort(converged_orientation_errors)[:5]
        for i, idx in enumerate(lowest_orientation_indices):
            print(f"  {i+1}. Sample {idx}: {converged_orientation_errors[idx]:.6f}")

# Usage
if __name__ == "__main__":
    file_path = "visual_servoing_data.npz"  # Replace with your file path if different
    print_npz_contents(file_path)

import os
import shutil
import xml.etree.ElementTree as ET
import torch
from torchvision.transforms import Compose, RandomErasing, ColorJitter
import cv2
import numpy as np

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

class TorchAugmentor(object):
    def __init__(self, composer):
        self.composer = composer

    def __call__(self, I):
        Itorch = torch.from_numpy(I).float().clone() / 255
        Itorch = Itorch.permute(2, 0, 1).unsqueeze(0)
        Itorch = self.composer(Itorch[0])
        Ip = Itorch.permute(1, 2, 0).unsqueeze(0) * 255
        return Ip.cpu().numpy()

def create_perturbed_model(source_model_path, dest_model_path, perturbed_image, model_number):
    # Copy the entire model folder
    shutil.copytree(source_model_path, dest_model_path, dirs_exist_ok=True)

    # Replace resized.png in materials/textures and meshes
    for subfolder in ['materials/textures', 'meshes']:
        image_path = os.path.join(dest_model_path, subfolder, 'resized.png')
        cv2.imwrite(image_path, cv2.cvtColor(perturbed_image.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Update model.sdf
    sdf_path = os.path.join(dest_model_path, 'model.sdf')
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    # Update the mesh URI and model name in model.sdf
    mesh_uri = root.find(".//uri")
    if mesh_uri is not None:
        new_uri = f"model://{os.path.basename(dest_model_path)}/meshes/resized.dae"
        mesh_uri.text = new_uri

    model_elem = root.find("model")
    if model_elem is not None:
        model_elem.set('name', f'resized{model_number}')

    tree.write(sdf_path)

    # Update model.config
    config_path = os.path.join(dest_model_path, 'model.config')
    tree = ET.parse(config_path)
    root = tree.getroot()

    # Update the model name in model.config
    name_elem = root.find('name')
    if name_elem is not None:
        name_elem.text = f'resized{model_number}'

    tree.write(config_path)

def main():
    # Set the random seed
    seed = 489
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define the augmentation pipeline
    augmentation = Compose([
        RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        ColorJitter(brightness=0.6, contrast=0.4),
        AddGaussianNoise(0.0, 0.05),
    ])

    # Create the augmentor
    augmentor = TorchAugmentor(augmentation)

    # Set paths
    source_model_path = 'models/viso'
    original_image_path = os.path.join(source_model_path, 'materials', 'textures', 'resized.png')

    # Load the original image
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        print(f"Error: Unable to load image {original_image_path}")
        return

    # Ensure the image is in RGB format
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Generate 500 perturbed models
    for i in range(1, 501):
        dest_model_path = f'models/viso{i}'

        # Generate perturbed image
        perturbed_image = augmentor(image_rgb)[0]

        # Create perturbed model
        create_perturbed_model(source_model_path, dest_model_path, perturbed_image, i)

        print(f"Created perturbed model: {dest_model_path}")

if __name__ == "__main__":
    main()

import os
import torch
from PIL import Image

class MovingObjectsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get the list of folder names in the root directory
        self.folder_names = [i for i in os.listdir(self.root_dir) if 'video_' in i]
        
    def __len__(self):
        # Return the number of folders in the root directory
        return len(self.folder_names)
    
    def __getitem__(self, index):
        # Get the folder name corresponding to the given index
        folder_name = self.folder_names[index]
        
        # Get the list of image filenames in the folder
        image_filenames = [i for i in os.listdir(os.path.join(self.root_dir, folder_name)) 
                           if i.endswith('.png')]
        image_filenames.sort(key= lambda i: int(i.lstrip('image_').rstrip('.png')))
        
        # Load the input images and target images into separate tensors
        input_images = []
        target_images = []
        for i, image_filename in enumerate(image_filenames):
            image_path = os.path.join(self.root_dir, folder_name, f"image_{i}.png")
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            if i < 11:
                # print(f"{image_filename} going in input")
                input_images.append(image)
            else:
                # print(f"{image_filename} going in target")
                target_images.append(image)
        
        # Convert the input and target image lists to tensors
        input_tensor = torch.stack(input_images)
        target_tensor = torch.stack(target_images)
        
        return input_tensor, target_tensor

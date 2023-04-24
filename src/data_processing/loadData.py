import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

# Dataset class
class HMEDataset(Dataset):
    """Dataset class for HME (Handwritten Math Expressions) dataset.

    Args:
        labels_file (str): File path to the CSV file containing image labels.
        images_dir (str): Directory path containing the images.
        transform (callable, optional): Optional transform to be applied on the images. Default is None.
        target_transform (callable, optional): Optional transform to be applied on the labels. Default is None.

    Returns:
        tuple: A tuple containing the following elements:
            - image (torch.Tensor): Tensor representing the image.
            - label (int or any other data type): Label associated with the image.
            - idx (int): Index of the image in the dataset.

    Examples:
        >>> dataset = HMEDataset('labels.csv', 'images/')
        >>> image, label, idx = dataset[0]
    """
    def __init__(self, labels_file, images_dir, offset=0, transform=None, target_transform=None):
        """
        Constructor method for HMEDataset.

        Args:
            labels_file (str): File path to the CSV file containing image labels.
            images_dir (str): Directory path containing the images.
            transform (callable, optional): Optional transform to be applied on the images. Default is None.
            target_transform (callable, optional): Optional transform to be applied on the labels. Default is None.
        """
        self.labels_file = pd.read_csv(labels_file, header=None)
        self.images_dir = images_dir
        self.offset = offset
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.labels_file)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the following elements:
                - image (torch.Tensor): Tensor representing the image.
                - label (int or any other data type): Label associated with the image.
                - idx (int): Index of the image in the dataset.
        """
        filename = 'iso' + str(idx + self.offset) + '.png'
        image_path = os.path.join(self.images_dir, filename)
        label = self.labels_file.iloc[idx, 1]
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label, idx

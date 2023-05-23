import os
import pickle

import h5py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

__all__ = [
    "TheatreDataLoaderPFG", "TheatreDataset"
]


class TheatreDataLoaderPFG(DataLoader):
    """
    Prefetch version of DataLoader: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class TheatreDataset(Dataset):
    """Images are loaded from by open specific file
    """

    @staticmethod
    def collate_fn(batch):
        """Use in torch.utils.data.DataLoader
        """

        return tuple(zip(*batch))  # as tuples instead of stacked tensors

    @staticmethod
    def get_transform():
        """More complicated transform utils in torchvison/references/detection/transforms.py
        """

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return transform

    def __init__(self, img_dir_root, theatre_data_path, look_up_tables_path, dataset_type=None,
                 transform=None):

        assert dataset_type in {None, 'train', 'test', 'val'}

        super(TheatreDataset, self).__init__()

        self.img_dir_root = img_dir_root
        self.theatre_data_path = theatre_data_path
        # self.depth_img_dir_root = depth_img_dir_root
        self.look_up_tables_path = look_up_tables_path
        self.dataset_type = dataset_type  # if dataset_type is None, all data will be used
        self.transform = transform

        # === load data here ====
        self.look_up_tables = pickle.load(open(look_up_tables_path, 'rb'))
    # we retain on whole data
    # def set_dataset_type(self, dataset_type, verbose=True):
    #
    #     assert dataset_type in {None, 'train', 'test', 'val'}
    #
    #     if verbose:
    #         print('[TheatreDataset]: {} switch to {}'.format(self.dataset_type, dataset_type))
    #
    #     self.dataset_type = dataset_type

    def __getitem__(self, idx):

        with h5py.File(self.theatre_data_path, 'r') as theatre_data:

            theatre_idx = idx

            img_path = os.path.join(self.img_dir_root, self.look_up_tables['idx_to_directory'][theatre_idx],
                                    self.look_up_tables['idx_to_filename'][theatre_idx])


            img = Image.open(img_path).convert("RGB")

            if self.transform is not None:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
          

            first_region_idx = theatre_data['img_to_first_box'][theatre_idx]
            last_region_idx = theatre_data['img_to_last_box'][theatre_idx]

            regions = torch.as_tensor(theatre_data['boxes'][first_region_idx: last_region_idx + 1],
                                      dtype=torch.float32)
            caps = torch.as_tensor(theatre_data['captions'][first_region_idx: last_region_idx + 1], dtype=torch.long)
            caps_len = torch.as_tensor(theatre_data['lengths'][first_region_idx: last_region_idx + 1], dtype=torch.long)

            targets = {
                'boxes': regions,
                'caps': caps,
                'caps_len': caps_len,
            }

            info = {
                'idx': theatre_idx,
                'dir': self.look_up_tables['idx_to_directory'][theatre_idx],
                'file_name': self.look_up_tables['idx_to_filename'][theatre_idx]
            }

        return img, targets, info
    
    
    
    def __len__(self):
      return len(self.look_up_tables['filename_to_idx'])




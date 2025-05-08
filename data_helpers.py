import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def pad_to_128(x):
    # Target shape: 128, 128, 128
    target_D, target_H, target_W = 128, 128, 128
    curr_D, curr_H, curr_W = x.shape[1:]

    pad_D = target_D - curr_D  # 128 - 81 = 47
    pad_H = target_H - curr_H  # 128 - 104 = 24
    pad_W = target_W - curr_W  # 128 - 83 = 45

    # Compute symmetric (or near-symmetric) padding
    pad_front = pad_D // 2           # 23
    pad_back = pad_D - pad_front     # 24
    pad_top = pad_H // 2             # 12
    pad_bottom = pad_H - pad_top     # 12
    pad_left = pad_W // 2            # 22
    pad_right = pad_W - pad_left     # 23

    # Apply padding
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
    return padded

class BrainImageSubjectDataset(Dataset):
    def __init__(self, fmri_data,image_data,labels,fmri_type,transform=None):
        super().__init__()
        self.fmri_data=fmri_data
        self.image_data=image_data
        self.labels=labels
        self.transform=transform
        self.fmri_type=fmri_type

    def __len__(self):
        return len(self.fmri_data)
    
    def __getitem__(self, idx):
        fmri = torch.tensor(self.fmri_data[idx])
        if self.fmri_type=="voxel":
            fmri=pad_to_128(fmri)
        image = torch.tensor(self.image_data[idx])
        label = torch.tensor(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'fmri': fmri,
            'label': label
        }
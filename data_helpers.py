import torch
from torch.utils.data import Dataset, DataLoader

class BrainImageSubjectDataset(Dataset):
    def __init__(self, fmri_data,image_data,labels,transform=None):
        super().__init__()
        self.fmri_data=fmri_data
        self.image_data=image_data
        self.labels=labels
        self.transform=transform

    def __len__(self):
        return len(self.fmri_data)
    
    def get_shapes(self):
        return self.fmri_data[0].shape,self.image_data[0].shape, self.labels[0].sha[e]
    
    def __getitem__(self, idx):
        fmri = torch.tensor(self.fmri_data[idx])
        image = torch.tensor(self.image_data[idx])
        label = torch.tensor(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'fmri': fmri,
            'label': label
        }
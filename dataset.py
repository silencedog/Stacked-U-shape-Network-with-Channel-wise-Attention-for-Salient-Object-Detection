import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms


class ImageData_train(data.Dataset):

    def __init__(self, img_root, label_root, transform, t_transform, filename=None):
        if filename is None:
            self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
            self.label_path = list(
                map(lambda x: os.path.join(label_root, x.split('/')[-1][:-3] + 'png'), self.image_path))
        else:
            lines = [line.rstrip('\n')[:-3] for line in open(filename)]
            self.image_path = list(map(lambda x: os.path.join(img_root, x + 'jpg'), lines))
            self.label_path = list(map(lambda x: os.path.join(label_root, x + 'png'), lines))

        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        label = Image.open(self.label_path[item]).convert('L')
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        
        return image, label

    def __len__(self):
        return len(self.image_path)

    
class ImageData_test(data.Dataset):

    def __init__(self, img_root, label_root, transform, t_transform, filename=None):
        if filename is None:
            self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
            self.label_path = list(
                map(lambda x: os.path.join(label_root, x.split('/')[-1][:-3] + 'png'), self.image_path))
        else:
            lines = [line.rstrip('\n')[:-3] for line in open(filename)]
            self.image_path = list(map(lambda x: os.path.join(img_root, x + 'jpg'), lines))
            self.label_path = list(map(lambda x: os.path.join(label_root, x + 'png'), lines))

        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        label = Image.open(self.label_path[item]).convert('L')
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        
        return image, label, self.label_path[item].split('/')[-1].split(".")[0]

    def __len__(self):
        return len(self.image_path)    
    
    

# get the dataloader (Note: without data augmentation)
def get_loader(img_root, label_root, img_size, batch_size, filename=None, mode='train', num_thread=4, pin=True):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        dataset = ImageData_train(img_root, label_root, transform, t_transform, filename=filename)
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                      pin_memory=pin)
        return data_loader
    else:
        t_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        dataset = ImageData_test(img_root, label_root, None, t_transform, filename=filename)
        return dataset


if __name__ == '__main__':
    import numpy as np
    img_root   = '/home/panzefeng/All_code/BDCN_salient_detection/DATASET/DUTS-TR/images'
    label_root = '/home/panzefeng/All_code/BDCN_salient_detection/DATASET/DUTS-TR/annotation'
    filename   = '/home/panzefeng/All_code/BDCN_salient_detection/DATASET/DUTS-TR/annotation_path.txt'
    loader     = get_loader(img_root, label_root, 256, 1, filename=filename, mode='train')
    for image, label in loader:
        print(np.array(label).shape)
        break

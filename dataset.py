import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data.sampler import Sampler

class ImageFolderWithName(datasets.ImageFolder):
    def __init__(self, root='./data', phase='train', shape=512, *args, **kwargs):
        self.transforms = transforms.Compose([
            transforms.Resize(shape),
            transforms.RandomCrop((shape, shape)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        assert phase in ['train', 'val', 'test']
        root = os.path.join(root, phase)
        super().__init__(root=root, transform=self.transforms, *args, **kwargs)
        self.return_fn = (phase == 'test') or (phase == 'val')

    def __getitem__(self, i):
        img, label = super(ImageFolderWithName, self).__getitem__(i)
        assert label <= 98*2
        if not self.return_fn:
            return img, label
        else:
            return img, label, self.imgs[i]

class CustomSampler(Sampler):
    def __init__(self, dataset, batch_size=32, batch_k=4, len=5000):
        assert batch_size % batch_k == 0
        self.batch_size = batch_size
        self.batch_k = batch_k
        self.len = len
        self.classes_per_batch = int(batch_size / batch_k)
        self.labels = np.array([data[1] for data in dataset.imgs])
        self.unique_labels = np.unique(self.labels)

    def __iter__(self):
        count = 0
        for i in range(len(self.labels)):
            class_ids = np.random.choice(self.unique_labels, self.classes_per_batch, replace=False)
            indices = []
            for label in class_ids:
                indices.extend(np.random.choice(np.nonzero(self.labels == label)[0], self.batch_k, replace=False))
            yield indices
            count += 1
            if count >= self.len:
                break

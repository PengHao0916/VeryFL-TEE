# dataset/CIFAR10.py (完整替换)

from torchvision import datasets, transforms
from config.dataset import dataset_file_path

def get_cifar10(train:bool = True):
    # CIFAR10 数据集的标准均值和标准差
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    return datasets.CIFAR10(root = dataset_file_path,
                                train = train,
                                download = True,
                                transform = transforms.Compose([
                                    transforms.RandomCrop(32,padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,  # <-- 在这里新增归一化
                                ]))
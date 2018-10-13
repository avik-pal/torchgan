import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets

def _get_transforms(output_shape=None):
    transformation = []
    if output_shape is not None:
        transformation.append(transforms.Resize(output_shape))
    transformation.append(transforms.ToTensor())
    transformation.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    return transformation

def mnist_dataloader(batch_size=128, download=True, train=True, shuffle=True,
                     root=".", output_shape=None, transformation=None):
    if transformation is None:
        transformation = _get_transforms(output_shape)
    dataset = dsets.MNIST(root=root, train=train,
                          transform=transforms.Compose(transformation),
                          download=download)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def cifar10_dataloader(batch_size=128, download=True, train=True, shuffle=True,
                       root=".", output_shape=None):
    if transformation is None:
        transformation = _get_transforms(output_shape)
    dataset = dsets.CIFAR10(root=root, train=train,
                            transform=transforms.Compose(transformation),
                            download=download)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def cifar100_dataloader(batch_size=128, download=True, train=True, shuffle=True,
                        root=".", output_shape=None):
    if transformation is None:
        transformation = _get_transforms(output_shape)
    dataset = dsets.CIFAR100(root=root, train=train,
                             transform=transforms.Compose(transformation),
                             download=download)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def fashionmnist_dataloader(batch_size=128, download=True, train=True, shuffle=True,
                            root=".", output_shape=None):
    if transformation is None:
        transformation = _get_transforms(output_shape)
    dataset = dsets.FashionMNIST(root=root, train=train,
                                 transform=transforms.Compose(transformation),
                                 download=download)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def emnist_dataloader(batch_size=128, download=True, train=True, shuffle=True,
                      root=".", output_shape=None):
    if transformation is None:
        transformation = _get_transforms(output_shape)
    dataset = dsets.EMNIST(root=root, train=train,
                           transform=transforms.Compose(transformation),
                           download=download)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def lsun_dataloader(batch_size=128, download=True, train=True, shuffle=True,
                    root=".", output_shape=None):
    if transformation is None:
        transformation = _get_transforms(output_shape)
    if download:
        raise ValueException("LSUN needs to be manually downloaded")
    dataset = dsets.LSUN(root=root, train=train,
                         transform=transforms.Compose(transformation))
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def stl10_dataloader(batch_size=128, download=True, train=True, shuffle=True,
                     root=".", output_shape=None):
    if transformation is None:
        transformation = _get_transforms(output_shape)
    dataset = dsets.STL10(root=root, train=train,
                          transform=transforms.Compose(transformation),
                          download=download)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def svhn_dataloader(batch_size=128, download=True, train=True, shuffle=True,
                     root=".", output_shape=None):
    if transformation is None:
        transformation = _get_transforms(output_shape)
    dataset = dsets.SVHN(root=root, train=train,
                         transform=transforms.Compose(transformation),
                         download=download)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

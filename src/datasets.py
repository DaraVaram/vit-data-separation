from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class DatasetLoader:
    def __init__(self, dataset_name, batch_size, use_percentage=1, train_percentage=0.9):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.use_percentage = use_percentage
        self.train_percentage = train_percentage
        self.test_percentage = 1 - train_percentage
        self.transform = None
        self.img_size = None
        self.num_classes = None
        self.train_loader = None
        self.test_loader = None
        
        self.set_transformations()
        
        self.load_data()

    def set_transformations(self):
        if 'CIFAR' in self.dataset_name:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.img_size = 32
            self.num_classes = 100 if self.dataset_name == 'CIFAR100' else 10
        elif 'MNIST' in self.dataset_name:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.img_size = 28
            self.num_classes = 10
        else:
            raise ValueError(f"Dataset {self.dataset_name} transformations is not supported.")

    def load_data(self):
        if self.dataset_name == 'MNIST':
            dataset = datasets.MNIST(root='data', train=True, transform=self.transform, download=True)
            test_dataset = datasets.MNIST(root='data', train=False, transform=self.transform, download=True)
        elif self.dataset_name == 'FashionMNIST':
            dataset = datasets.FashionMNIST(root='data', train=True, transform=self.transform, download=True)
            test_dataset = datasets.FashionMNIST(root='data', train=False, transform=self.transform, download=True)
        elif self.dataset_name == 'CIFAR10':
            dataset = datasets.CIFAR10(root='data', train=True, transform=self.transform, download=True)
            test_dataset = datasets.CIFAR10(root='data', train=False, transform=self.transform, download=True)
        elif self.dataset_name == 'CIFAR100':
            dataset = datasets.CIFAR100(root='data', train=True, transform=self.transform, download=True)
            test_dataset = datasets.CIFAR100(root='data', train=False, transform=self.transform, download=True)
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")
        
        # Calculate sizes
        total_size = int(len(dataset) * self.use_percentage)
        train_size = int(total_size * self.train_percentage)
        test_size = total_size - train_size
        
        remaining_size = len(dataset) - total_size
        subset_dataset, _ = random_split(dataset, [total_size, remaining_size])
        train_dataset, test_dataset = random_split(subset_dataset, [train_size, test_size])

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader
    
    def get_dataset_info(self):
        return {
            "dataset_name": self.dataset_name,
            "image_size": self.img_size,
            "num_classes": self.num_classes
        }
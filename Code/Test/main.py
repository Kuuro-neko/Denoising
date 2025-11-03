import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.models import resnet50


#Paths
train_noisy_dir = "./archive/train/train/noisy images"
train_ground_truth_dir = "./archive/train/train/ground truth"
val_noisy_dir = "./archive/validate/validate/noisy images"
val_ground_truth_dir = "./archive/validate/validate/ground truth"
test_noisy_dir = "./archive/test/test/noisy images"
test_ground_truth_dir = "./archive/test/test/ground truth"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(16 * 64 * 64, 64)  # Adjust dimensions as needed
        self.fc2 = nn.Linear(64, 64)  # Adjust output classes as needed

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.max_pool2d(x, 4)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, label) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        label = torch.tensor(label).to(device)  # Convert label to Tensor and move to device
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, _, target in test_loader:  # Ignore ground_truth_image with `_`
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Custom Dataset
class NoisyImageDataset(Dataset):
    def __init__(self, noisy_dir, ground_truth_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.labels = [img.split('_')[0] for img in self.noisy_images]  # Extract class from filename

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        noise_types = ["salt and pepper", "speckle", "poisson", "gauss"]
        ground_truth_name = self.noisy_images[idx]
        for noise_type in noise_types:
            ground_truth_name = ground_truth_name.replace(f"{noise_type}_", "")
        ground_truth_path = os.path.join(self.ground_truth_dir, ground_truth_name)

        noisy_image = Image.open(noisy_path).convert("RGB")
        ground_truth_image = Image.open(ground_truth_path).convert("RGB")

        if self.transform:
            noisy_image = self.transform(noisy_image)
            ground_truth_image = self.transform(ground_truth_image)

        # Convert string label to integer using the mapping
        label = self.labels[idx]
        label_idx = noise_types.index(label)

        return noisy_image, ground_truth_image, torch.tensor(label_idx)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")


    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 4,
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Datasets and DataLoaders
    train_dataset = NoisyImageDataset(train_noisy_dir, train_ground_truth_dir, transform)
    test_dataset = NoisyImageDataset(test_noisy_dir, test_ground_truth_dir, transform)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    # Model
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        torch.cuda.empty_cache()
        test(model, device, test_loader)
        scheduler.step()


if __name__ == '__main__':
    main()

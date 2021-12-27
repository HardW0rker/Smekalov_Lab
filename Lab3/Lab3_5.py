import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from torch import nn
from torchvision.utils import make_grid
from torchvision import datasets, models, transforms
import torch.utils.data as tdata
from sklearn import model_selection
from IPython.display import display
import time
import os

script_dir = os.path.dirname(__file__)
folder_path = os.path.join(script_dir, 'intel')
train_path = folder_path + '/seg_train/seg_train'
test_path = folder_path + '/seg_test/seg_test'

data_path_format = folder_path + '/seg_{0}/seg_{0}'

np.random.seed(5315)
torch.manual_seed(9784)
channel_means = (0.485, 0.456, 0.406)
channel_stds = (0.229, 0.224, 0.225)

image_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(channel_means, channel_stds)
])

image_datasets = dict(zip(('dev', 'test'),[datasets.ImageFolder(data_path_format.format(key), transform=image_transforms) for key in['train', 'test']]))

devset_indices = np.arange(len(image_datasets['dev']))
devset_labels = image_datasets['dev'].targets

train_indices, val_indices, train_labels, val_labels = model_selection.train_test_split(devset_indices, devset_labels,test_size=0.1,stratify=devset_labels)

image_datasets['train'] = tdata.Subset(image_datasets['dev'], train_indices)
image_datasets['validation'] = tdata.Subset(image_datasets['dev'], val_indices)

image_dataloaders = {key: tdata.DataLoader(image_datasets[key], batch_size=16, shuffle=True) for key in ['train', 'validation']}
image_dataloaders['test'] = tdata.DataLoader(image_datasets['test'], batch_size=32)


def imshow(inp, title=None, fig_size=None):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = channel_stds * inp + channel_means
    inp = np.clip(inp, 0, 1)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot('111')
    ax.imshow(inp)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect('equal')
    plt.pause(0.001)


cuda_device = torch.device('cuda') 
#cpu_device = torch.device('cpu')
device = cuda_device

ptr = models.resnet18(pretrained=True)
for param in ptr.parameters():
    param.requires_grad = False
num_ftrs = ptr.fc.in_features
ptr.fc = nn.Linear(num_ftrs, 6)
ptr = ptr.to(device)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(256 * 34 * 34, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.ReLU()
        )

    def forward(self, xb):
        return self.network(xb)


def train_model(epochs, model, optimizer, criterion, loaders, device, best_model, n_prints=5):
    print_every = len(loaders['train']) // n_prints
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for iteration, (xx, yy) in enumerate(loaders['train']):
            optimizer.zero_grad()
            xx, yy = xx.to(device), yy.to(device)
            out = model(xx)
            loss = criterion(out, yy)
            running_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (iteration % print_every == print_every - 1):
                running_train_loss /= print_every
                print(f"Epoch {epoch}, iteration {iteration} training_loss {running_train_loss}")
                running_train_loss = 0.0

        with torch.no_grad():
            model.eval()
            running_corrects = 0
            running_total = 0
            running_loss = 0.0
            for xx, yy in loaders['validation']:
                batch_size = xx.size(0)
                xx, yy = xx.to(device), yy.to(device)

                out = model(xx)

                loss = criterion(out, yy)
                running_loss += loss.item()

                predictions = out.argmax(1)
                running_corrects += (predictions == yy).sum().item()
                running_total += batch_size

            mean_val_loss = running_loss / len(loaders['validation'])
            accuracy = running_corrects / running_total

            print(f"Epoch {epoch}, val_loss {mean_val_loss}, accuracy = {accuracy}")


start = time.time()
optimizer = torch.optim.Adam(ptr.fc.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()
best_model = Model()
train_model(1, ptr, optimizer, criterion, image_dataloaders, device, best_model=best_model, n_prints=5)

end = time.time()
print(end - start)

xx, yy = next(iter(image_dataloaders['validation']))

all_preds = []
correct_preds = []
with torch.no_grad():
    ptr.eval()
    for xx, yy in image_dataloaders['test']:
        xx = xx.to(device)
        output = ptr(xx)
        all_preds.extend(output.argmax(1).tolist())
        correct_preds.extend(yy.tolist())

all_preds = np.asarray(all_preds)
correct_preds = np.asarray(correct_preds)

target_names = image_datasets['test'].classes
print(metrics.classification_report(correct_preds, all_preds, target_names=target_names))

confusion_matrix = metrics.confusion_matrix(correct_preds, all_preds)
print(pd.DataFrame(confusion_matrix, index=target_names, columns=target_names))
#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
import neural_network_architectures as NNA
#-----------------------------------------------------------------------------------------#
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#-----------------------------------------------------------------------------------------#
from torch.utils.data import DataLoader, Dataset, random_split
from torchsummary import summary
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
#-----------------------------------------------------------------------------------------#

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx][0]
        label = self.data[idx][1]
        data = np.load(data_path)
        data = torch.from_numpy(data)
        return data, label

data_dir = '../facies' # where is the main folder containing subclasses
train_size = 0.75; val_size = 0.2; test_size = 0.05
batch_size = 32
image_dim_1 = 33; image_dim_2 = 33
output_dim = 9 # should equal to number of classes
learning_rate = 0.001

class_names = os.listdir(data_dir)
print('class names: ', class_names)
num_class = len(class_names)
image_files = glob.glob(data_dir + '/*/*.npy', recursive=True)
print('total images in: ', data_dir, ' is ', len(image_files))

idx_to_class = {i: j for i, j in enumerate(class_names)}
class_to_idx = {value: key for key, value in idx_to_class.items()}

train_idx, val_idx, test_idx = random_split(image_files, [train_size, val_size, test_size])
train_list = [image_files[i] for i in train_idx.indices]
val_list = [image_files[i] for i in val_idx.indices]
test_list = [image_files[i] for i in test_idx.indices]

train_data = CustomDataset([(f,
	class_to_idx[os.path.basename(os.path.dirname(f))]) for f in train_list])
val_data = CustomDataset([(f,
    class_to_idx[os.path.basename(os.path.dirname(f))]) for f in val_list])
test_data = CustomDataset([(f,
    class_to_idx[os.path.basename(os.path.dirname(f))]) for f in test_list])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

print('number of training images: ', len(train_list),
      '\nnumber of val images: ', len(val_list),
      '\nnumber of test images: ', len(test_list))

# import matplotlib.pyplot as plt

# # Select a random subset of the training set
# random_train_indices = np.random.choice(len(train_data), size=25, replace=False)

# # Display the selected images with their labels
# fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10,10))
# for i, idx in enumerate(random_train_indices):
#     data, label = train_data[idx]
#     row, col = i // 5, i % 5
#     dummy = data.squeeze().numpy()
#     # print(dummy.shape)
#     axes[row, col].imshow(data.squeeze().numpy(), cmap='rainbow')
#     axes[row, col].set_title(class_names[label])
#     axes[row, col].axis('off')
# plt.show()

model = NNA.AlexNet(output_dim)
print(model.to(device))
summary(model, (1, image_dim_1, image_dim_2))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)



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
import time
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchsummary import summary
from tqdm import tqdm
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
#-----------------------------------------------------------------------------------------#

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_dim_1, image_dim_2)),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx][0]
        label = self.data[idx][1]
        with Image.open(data_path) as img:
            img_arr = np.asarray(img.convert('RGB')).copy()
        data = self.transform(img_arr)
        return data, label

data_dir = '../facies' # where is the main folder containing subclasses
train_size = 0.75; val_size = 0.20; test_size = 0.05
batch_size = 32
image_dim_1 = 33; image_dim_2 = 33
output_dim = 9 # should equal to number of classes
learning_rate = 1e-7
epochs = 1000
save_model = '../trained_models/ResNet32.pt'
model_name = 'ResNet32'
lithofacies = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
ResNet_achitecture = 'ResNet32'

class_names = os.listdir(data_dir)
print('class names: ', class_names)
num_class = len(class_names)
image_files = glob.glob(data_dir + '/*/*.png', recursive=True)
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


# Select a random subset of the training set
# random_train_indices = np.random.choice(len(train_data), size=25, replace=False)

# # Display the selected images with their labels
# fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10,10))
# for i, idx in enumerate(random_train_indices):
#     data, label = train_data[idx]
#     row, col = i // 5, i % 5
#     if data.shape[-1] == 1:
#         # grayscale image
#         img_arr = np.stack([data.squeeze().numpy()] * 3, axis=-1)
#     else:
#         # RGB image
#         img_arr = data.permute(1, 2, 0).numpy()
#     axes[row, col].imshow(img_arr, cmap='rainbow')
#     axes[row, col].set_title(class_names[label])
#     axes[row, col].axis('off')
# plt.show()

'''
step 3: Model Initialization and Parameter Selection.

Architecture choices: ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202 
'''

ResNet_config = U.ResNet_achitecture_choices(ResNet_achitecture)
model = NNA.ResNet(ResNet_config, output_dim)
print(model.to(device))
summary(model, (3, image_dim_1, image_dim_2))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)
model = model.to(device)

history_train_acc = []
history_valid_acc = []
best_valid_loss = float('inf')
for epoch in range(epochs):
	start_time = time.monotonic()
	train_loss, train_acc = U.train_3_channel(model, train_loader, optimizer, criterion, device)
	valid_loss, valid_acc = U.evaluate_3_channel(model, val_loader, criterion, device)
	history_train_acc.append(train_acc)
	history_valid_acc.append(valid_acc)
	if valid_loss < best_valid_loss:
		best_valid_loss = valid_loss
		torch.save(model.state_dict(), save_model)
	end_time = time.monotonic()
	epoch_mins, epoch_secs = U.epoch_time(start_time, end_time)
	print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
	print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
	print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
U.loss_history_plot(history_train_acc, history_valid_acc, model_name)

model.load_state_dict(torch.load(save_model))
test_loss, test_acc = U.evaluate_3_channel(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
images, labels, probs = U.get_predictions_3_channel(model, test_loader, device)
pred_labels = torch.argmax(probs, 1)
U.plot_confusion_matrix_tabular(labels, pred_labels, lithofacies)


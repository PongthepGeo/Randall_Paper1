#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
import neural_network_architectures as NNA
#-----------------------------------------------------------------------------------------#
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
#-----------------------------------------------------------------------------------------#
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from tqdm import tqdm
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
#-----------------------------------------------------------------------------------------#

'''
step 0:
'''

data_dir = '../facies' # where is the main folder containing subclasses
train_size = 0.6; val_size = 0.35; test_size = 0.05
batch_size = 512
image_dim_1 = 33; image_dim_2 = 33
output_dim = 9 # should equal to number of classes
learning_rate = 1e-5
epochs = 50
model_name = 'ResNet32'
save_model = '../trained_models/' + model_name + '.pt'
lithofacies = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']

'''
step 1:
'''

# NOTE
class_names = os.listdir(data_dir)
print('class names: ', class_names)
num_class = len(class_names)
image_files = glob.glob(data_dir + '/*/*.png', recursive=True)
print('total images in: ', data_dir, ' is ', len(image_files))
# NOTE
idx_to_class = {i: j for i, j in enumerate(class_names)}
class_to_idx = {value: key for key, value in idx_to_class.items()}
train_idx, val_idx, test_idx = random_split(image_files, [train_size, val_size, test_size])
train_list = [image_files[i] for i in train_idx.indices]
val_list = [image_files[i] for i in val_idx.indices]
test_list = [image_files[i] for i in test_idx.indices]
# NOTE
train_data = NNA.FaciesWellLog([(f,
    class_to_idx[os.path.basename(os.path.dirname(f))]) for f in train_list],
    image_dim_1, image_dim_2)
val_data = NNA.FaciesWellLog([(f,
    class_to_idx[os.path.basename(os.path.dirname(f))]) for f in val_list],
    image_dim_1, image_dim_2)
test_data = NNA.FaciesWellLog([(f,
    class_to_idx[os.path.basename(os.path.dirname(f))]) for f in test_list],
    image_dim_1, image_dim_2)
# NOTE
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
# NOTE
print('number of training images: ', len(train_list),
      '\nnumber of val images: ', len(val_list),
      '\nnumber of test images: ', len(test_list))
# NOTE
# U.preview_well_logs(train_data, class_names)

'''
step 2:
'''

ResNet_config = U.ResNet_achitecture_choices(model_name)
model = NNA.ResNet(ResNet_config, output_dim)
print(model.to(device))
summary(model, (3, image_dim_1, image_dim_2))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)
model = model.to(device)

'''
step 3:
'''

history_train_acc = []
history_valid_acc = []
best_valid_loss = float('inf')
for epoch in range(epochs):
	start_time = time.monotonic()
	train_loss, train_acc = U.train2D(model, train_loader, optimizer, criterion, device)
	valid_loss, valid_acc = U.evaluate2D(model, val_loader, criterion, device)
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

'''
step 4:
'''

model.load_state_dict(torch.load(save_model))
test_loss, test_acc = U.evaluate2D(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
images, labels, probs = U.get_predictions2D(model, test_loader, device)
pred_labels = torch.argmax(probs, 1)
U.plot_confusion_matrix_tabular(labels, pred_labels, lithofacies)
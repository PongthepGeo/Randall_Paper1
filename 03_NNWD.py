#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
import neural_network_architectures as NNA
#-----------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import time
import torch.utils.data as data
#-----------------------------------------------------------------------------------------#
from sklearn.preprocessing import LabelEncoder
from torchsummary import summary
from tqdm import tqdm
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
#-----------------------------------------------------------------------------------------#

'''
step 0: Predifined Parameters
'''

batch_size = 16
val_ratio = 0.8
epochs = 200
learning_rate = 1e-5
model_name = 'NeuralNetWithDropout'
save_model = '../trained_models/' + model_name + '.pt'
lithofacies = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']

'''
step 1: Data Loading and Data Wrangling.

1) Load the well logs data from a CSV file using Pandas.
2) Replace missing values with -999.
3) Encode the facies column using LabelEncoder to convert string labels to integer labels.
4) Split the data into training and testing sets based on the "Well Name" column.
5) Drop unnecessary columns from the training and testing data.
6) Convert the training and testing data to NumPy arrays.
7) Convert the NumPy arrays to PyTorch tensors.
'''

# NOTE 1-3
well_logs = pd.read_csv('datasets/well_logs.csv')
well_logs.fillna(-999, inplace=True)
le = LabelEncoder()
well_logs['Facies'] = le.fit_transform(well_logs['Facies'])
# NOTE 4-5
test_well = 'CROSS H CATTLE'
train = well_logs.loc[well_logs['Well Name'] != test_well]
test = well_logs.loc[well_logs['Well Name'] == test_well]
drop_cols = ['Facies', 'Formation', 'Well Name', 'Depth'] 
X_train = train.drop(drop_cols, axis=1) 
y_train = train['Facies'] 
X_test = test.drop(drop_cols, axis=1) 
y_test = test['Facies']
# NOTE 6-7
X_train_np = X_train.values.astype('float32')
y_train_np = y_train.values.astype('int64')
X_test_np = X_test.values.astype('float32')
y_test_np = y_test.values.astype('int64')
X_train_tensor = torch.from_numpy(X_train_np)
y_train_tensor = torch.from_numpy(y_train_np)
X_test_tensor = torch.from_numpy(X_test_np)
y_test_tensor = torch.from_numpy(y_test_np)

'''
step 2: Creating Data Loaders and Datasets for Training, Validation, and Testing.

1) Determine the number of training examples to use for validation based on a specified validation ratio.
2) Use the 'random_split' function from the data module to split the training data into training and validation sets.
3) Create TensorDataset objects for the training, validation, and test data.
4) Create DataLoader objects for the training, validation, and test datasets. The batch_size argument specifies the number of samples to use in each batch, and the shuffle argument controls whether the order of the samples is randomized. The drop_last argument controls whether to drop the last batch if it contains fewer samples than the specified batch size.
'''

# NOTE 1
n_train_examples = int(len(X_train_tensor) * val_ratio)
n_valid_examples = len(X_train_tensor) - n_train_examples
# NOTE 2
train_data, val_data = data.random_split(data.TensorDataset(X_train_tensor, y_train_tensor),
	[n_train_examples, n_valid_examples])
# NOTE 3
train_dataset = data.TensorDataset(train_data.dataset.tensors[0],
								   train_data.dataset.tensors[1])
val_dataset = data.TensorDataset(val_data.dataset.tensors[0],
								 val_data.dataset.tensors[1])
test_dataset = data.TensorDataset(X_test_tensor, y_test_tensor)
# NOTE 4
train_loader = data.DataLoader(
	train_dataset,
	batch_size=batch_size,
	shuffle=True,
	drop_last=True
)
val_loader = data.DataLoader(
	val_dataset,
	batch_size=batch_size,
	shuffle=False,
	drop_last=True
)
test_loader = data.DataLoader(
	test_dataset,
	batch_size=batch_size,
	shuffle=False,
	drop_last=True
)

'''
step 3: Defining Neural Network with Optimization and Computing Loss.
'''

model = NNA.NNWD(input_dim=X_train.shape[1], output_dim=len(np.unique(y_train_np)))
model.to(device)
summary(model, (1, X_train.shape[1]))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)

'''
step 4: Model Training and Evaluation.
'''

history_train_acc = []
history_valid_acc = []
best_valid_loss = float('inf')
for epoch in range(epochs):
	start_time = time.monotonic()
	train_loss, train_acc = U.train1D(model, device, train_loader, optimizer, criterion)
	valid_loss, valid_acc = U.evaluate1D(model, device, val_loader, criterion)
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
step 5: Confusion Matrix.
'''

model.load_state_dict(torch.load(save_model))
test_loss, test_acc = U.evaluate1D(model, device, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
images, labels, probs = U.get_predictions1D(model, test_loader, device)
pred_labels = torch.argmax(probs, 1)
U.plot_confusion_matrix_tabular(labels, pred_labels, lithofacies)
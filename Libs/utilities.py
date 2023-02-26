#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import neural_network_architectures as NNA
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torch.nn.functional as TF
import time
#-----------------------------------------------------------------------------------------#
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_fscore_support
from matplotlib.patches import Patch
from collections import namedtuple
#-----------------------------------------------------------------------------------------#
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':11,  
	'axes.titlesize':11,
	'axes.titleweight': 'bold',
	'legend.fontsize': 11,  # was 10
	'xtick.labelsize':11,
	'ytick.labelsize':11,
	'font.family': 'serif',
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

def sub_cm(vector):
	dummy = []
	for i in range (0, len(vector)):
		if vector[i] == 0:
			dummy.append('SS')
		elif vector[i] == 1:
			dummy.append('CSiS')
		elif vector[i] == 2:
			dummy.append('FSiS')
		elif vector[i] == 3:
			dummy.append('SiSh')
		elif vector[i] == 4:
			dummy.append('MS')
		elif vector[i] == 5:
			dummy.append('WS')
		elif vector[i] == 6:
			dummy.append('D')
		elif vector[i] == 7:
			dummy.append('PS')
		elif vector[i] == 8:
			dummy.append('BS')
	return dummy

def cm(y_pred, data, selected_well, lithofacies):
	y_true = data.Facies.loc[data['Well Name'] == selected_well]
	y_true = y_true.to_numpy()
	y_true = sub_cm(y_true)
	y_pred = sub_cm(y_pred)
	weighted_f1 = f1_score(y_true, y_pred, average='weighted')
	dummy = confusion_matrix(y_true, y_pred, labels=lithofacies)
	disp = ConfusionMatrixDisplay(confusion_matrix=dummy, display_labels=lithofacies)
	disp.plot(cmap='Greens') 
	# plt.savefig('data_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()
	return weighted_f1

def difference(logs, pre, label):
	pre = pre[:, 0]
	# dummy = logs[label].to_numpy() - 1 # -1 is bug fixed to compensate number starting from 1 (not 0)
	dummy = logs[label].to_numpy() 
	diff = np.zeros(shape=(len(pre)), dtype=np.int8)
	# print(dummy)
	count = 0
	for i in range (0, len(pre)):
		if dummy[i] - pre[i] == 0:
			diff[i] = 1
			count += 1
	percent_diff = 1 - (count/len(pre))
	percent_diff = round(percent_diff, 4)
	return diff, percent_diff

def custom_metric(logs, label, predictions):
	lithocolors  = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00', '#1B4F72',
					'#2E86C1', '#AED6F1', '#A569BD', '#196F3D']
	logs = logs.sort_values(by='Depth', ascending=True)
	cmap = colors.ListedColormap(lithocolors)
	ztop = logs.Depth.min(); zbot=logs.Depth.max()
	# true = np.repeat(np.expand_dims(logs[label].values-1, 1), 5, 1) 
	true = np.repeat(np.expand_dims(logs[label].values, 1), 5, 1) 
	predictions = np.repeat(np.expand_dims(predictions, 1), 5, 1)
	f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 8))
	ax1.imshow(true, interpolation='none', aspect='auto', cmap=cmap, vmin=0, vmax=8, extent=[1, 5, zbot, ztop])
	ax2.imshow(predictions, interpolation='none', aspect='auto', cmap=cmap, vmin=0, vmax=8)

	diff, percent_diff = difference(logs, predictions, label)
	diff = np.repeat(np.expand_dims(diff, 1), 5, 1)
	cmap = colors.ListedColormap(['black', 'yellow'])
	ax3.imshow(diff, interpolation='none', aspect='auto', cmap=cmap, vmin=0, vmax=1)
	legend_elements = [Patch(facecolor='black', edgecolor='black', label='incorrect'),
					   Patch(facecolor='yellow', edgecolor='black', label='correct')]
	ax3.legend(handles=legend_elements, bbox_to_anchor=(1.9, 1.01),
			   framealpha=1, edgecolor='black')	

	for ax in f.get_axes():
		ax.label_outer()
	ax1.set_xticklabels([]); ax2.set_xticklabels([]); ax3.set_xticklabels([])
	ax1.set_xticks([]); ax2.set_xticks([]); ax3.set_xticks([])
	ax1.set_xlabel('True')
	ax2.set_xlabel('Prediction')
	ax3.set_xlabel('Difference')
	ax3.set_title('error: ' + str(percent_diff), loc='center')

	f.suptitle('Well: %s'%logs.iloc[0]['Well Name'])
	plt.tight_layout()
	# plt.savefig('data_out/' + save_file + '.svg', format='svg',
	#             bbox_inches='tight', transparent=True, pad_inches=0.1)
	plt.show()
	list_true_facies = (logs[label].sort_values(ascending=True)).unique()
	list_pre_facies = np.unique(predictions)
	return percent_diff, list_true_facies, list_pre_facies

def plot_loss_curves(results, evaluation_matrix):
	epochs = len(results['validation_0'][evaluation_matrix])
	x_axis = range(0, epochs)
	_, ax = plt.subplots(figsize=(6, 10))
	ax.plot(x_axis, results['validation_0'][evaluation_matrix], label='train')
	ax.plot(x_axis, results['validation_1'][evaluation_matrix], label='validation')
	ax.legend()
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.title('Multiclass Classification Error Rate')
	plt.show()

def plot_confusion_matrix(y_pred, data, test_well, y_test):
	lithofacies = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
	cm(y_pred, data, test_well, lithofacies)
	precision, recall, fscore, support = precision_recall_fscore_support(y_test,
	y_pred, average='weighted', zero_division=1)
	print('precision: ', precision,
		  '\nrecall: ', recall,
		  '\nf1 score: ', fscore,
		  '\nsupport: ', support
		  )

def plot_feature_importances(clf_xgb, data):
	plt.figure(figsize=(10, 6))
	plt.bar(range(len(clf_xgb.feature_importances_)), clf_xgb.feature_importances_)
	# print(len(clf_xgb.feature_importances_))
	drop_cols_2 = ['Facies', 'Formation', 'Well Name', 'Depth'] 
	new_data = data.drop(drop_cols_2, axis=1) 
	labels = new_data.columns[:]
	x = np.arange(0, len(labels), 1)
	plt.xticks(x, labels, rotation=90)
	plt.ylabel('values (the more is the better)')
	plt.title('Feature Importances')
	plt.tight_layout()
	plt.show()

def calculate_accuracy(y_pred, y):
	top_pred = y_pred.argmax(1, keepdim=True)
	correct = top_pred.eq(y.view_as(top_pred)).sum()
	acc = correct.float() / y.shape[0]
	return acc

def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs

def loss_history_plot(history_train, history_valid, model_name):
	axis_x = np.linspace(0, len(history_train), len(history_train))
	plt.plot(axis_x, history_train, linestyle='solid',
			 color='red', linewidth=1, marker='o', ms=5, label='train')
	plt.plot(axis_x, history_valid, linestyle='solid',
			 color='blue', linewidth=1, marker='o', ms=5, label='valid')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.legend(['train', 'valid'])
	plt.title(model_name + ': ' + 'Accuracy', fontweight='bold')
	plt.show()

def get_predictions1D(model, iterator, device):
	model.eval()
	images = []; labels = []; probs = []
	with torch.no_grad():
		for (x, y) in iterator:
			x = x.to(device)
			y_pred = model(x)
			y_prob = TF.softmax(y_pred, dim=-1)
			images.append(x.cpu())
			labels.append(y.cpu())
			probs.append(y_prob.cpu())
	images = torch.cat(images, dim=0)
	labels = torch.cat(labels, dim=0)
	probs = torch.cat(probs, dim=0)
	return images, labels, probs

def plot_confusion_matrix_tabular(labels, pred_labels, classes):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	cm = confusion_matrix(labels, pred_labels)
	cm = ConfusionMatrixDisplay(cm, display_labels=classes)
	cm.plot(values_format='d', cmap='Greens', ax=ax)
	plt.show()

def train1D(model, device, train_loader, optimizer, criterion):
	model.train()
	train_loss = 0
	train_acc = 0
	for X, y in tqdm(train_loader, desc='Training', leave=False, miniters=5):
		X, y = X.to(device), y.to(device)
		optimizer.zero_grad()
		output = model(X)
		loss = criterion(output, y)
		loss.backward()
		optimizer.step()
		train_loss += loss.item() * X.size(0)
		_, predicted = torch.max(output.data, 1)
		train_acc += (predicted == y).sum().item()
	train_loss /= len(train_loader.dataset)
	train_acc /= len(train_loader.dataset)
	return train_loss, train_acc

def evaluate1D(model, device, val_loader, criterion):
	model.eval()
	val_loss = 0
	val_acc = 0
	with torch.no_grad():
		for X, y in tqdm(val_loader, desc='Validation', leave=False, miniters=5):
			X, y = X.to(device), y.to(device)
			output = model(X)
			loss = criterion(output, y)
			val_loss += loss.item() * X.size(0)
			_, predicted = torch.max(output.data, 1)
			val_acc += (predicted == y).sum().item()
	val_loss /= len(val_loader.dataset)
	val_acc /= len(val_loader.dataset)
	return val_loss, val_acc

def train2D(model, iterator, optimizer, criterion, device):
	epoch_loss = 0; epoch_acc = 0
	model.train()
	for (x, y) in tqdm(iterator, desc='Training', leave=False):
		x = x.to(device)
		y = y.to(device)
		optimizer.zero_grad()
		y_pred, _ = model(x)
		loss = criterion(y_pred, y)
		acc = calculate_accuracy(y_pred, y)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
		epoch_acc += acc.item()
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate2D(model, iterator, criterion, device):
	epoch_loss = 0; epoch_acc = 0
	model.eval()
	with torch.no_grad():
		for (x, y) in tqdm(iterator, desc='Evaluating', leave=False):
			x = x.to(device)
			y = y.to(device)
			y_pred, _ = model(x)
			loss = criterion(y_pred, y)
			acc = calculate_accuracy(y_pred, y)
			epoch_loss += loss.item()
			epoch_acc += acc.item()
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def get_predictions2D(model, iterator, device):
	model.eval()
	images = []; labels = []; probs = []
	with torch.no_grad():
		for (x, y) in iterator:
			x = x.to(device)
			y_pred, _ = model(x)
			y_prob = TF.softmax(y_pred, dim=-1)
			images.append(x.cpu())
			labels.append(y.cpu())
			probs.append(y_prob.cpu())
	images = torch.cat(images, dim=0)
	labels = torch.cat(labels, dim=0)
	probs = torch.cat(probs, dim=0)
	return images, labels, probs

def ResNet_achitecture_choices(ResNet_achitecture):
	ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
	if ResNet_achitecture == 'ResNet20':
		n_blocks = [3, 3, 3]
		print('Using ResNet20')
	elif ResNet_achitecture == 'ResNet32':
		n_blocks = [5, 5, 5]
		print('Using ResNet32')
	elif ResNet_achitecture == 'ResNet44':
		n_blocks = [7, 7, 7]
		print('Using ResNet44')
	elif ResNet_achitecture == 'ResNet56':
		n_blocks = [9, 9, 9]
		print('Using ResNet56')
	elif ResNet_achitecture == 'ResNet110':
		n_blocks = [18, 18, 18]
		print('Using ResNet110')
	elif ResNet_achitecture == 'ResNet1202':
		n_blocks = [20, 20, 20]
		print('Using ResNet1202')
	else:
		print('out of ResNet architectures')
	return ResNetConfig(block = NNA.BasicBlock, n_blocks = n_blocks, channels = [16, 32, 64])

def preview_well_logs(train_data, class_names):
	random_train_indices = np.random.choice(len(train_data), size=16, replace=False)
	_, axes = plt.subplots(nrows=4, ncols=4, gridspec_kw={'hspace': 0.6})
	for i, idx in enumerate(random_train_indices):
		data, label = train_data[idx]
		row, col = i // 4, i % 4
		if data.shape[-1] == 1:
			# grayscale image
			img_arr = np.stack([data.squeeze().numpy()] * 3, axis=-1)
		else:
			# RGB image
			img_arr = data.permute(1, 2, 0).numpy()
		axes[row, col].imshow(img_arr, cmap='rainbow')
		axes[row, col].set_title(class_names[label])
		axes[row, col].axis('off')
	plt.show()
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#-----------------------------------------------------------------------------------------#
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_fscore_support
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
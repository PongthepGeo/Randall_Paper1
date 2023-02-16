#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
import xgboost as xgb
import numpy as np
import optuna
#-----------------------------------------------------------------------------------------#
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
#-----------------------------------------------------------------------------------------#

'''
step 1: Data Loading and Data Wrangling.
'''

data = pd.read_csv('datasets/well_logs.csv')
data.fillna(-999, inplace=True)
le = LabelEncoder()
data['Facies'] = le.fit_transform(data['Facies'])

'''
step 2: Creating Train, Validation, and Test Sets.
'''

test_well = 'CROSS H CATTLE'
train = data.loc[data['Well Name'] != test_well]
test = data.loc[data['Well Name'] == test_well]
drop_cols = ['Facies', 'Formation', 'Well Name', 'Depth'] 
X = train.drop(drop_cols, axis=1) 
y = train['Facies'] 
X_test = test.drop(drop_cols, axis=1) 
y_test = test['Facies'] 

'''
step 3: Hyperparameter Tuning. 
'''

def objective(trial, data=X, target=y):
	X_train, X_val, y_train, y_val = train_test_split(X, y, 
													  test_size=0.2,
													  random_state=True,
													  shuffle=y,
													  stratify=y)
	param = {
		# NOTE defines booster, gblinear for linear functions.
		# 'booster': trial.suggest_categorical('booster', ['gbtree']),
		'booster': 'gbtree',
		# NOTE L2 regularization weight.
		'lambda': trial.suggest_float('lambda', 1e-8, 1., log=True),
		# NOTE L1 regularization weight.
		'alpha': trial.suggest_float('alpha', 1e-8, 1., log=True),
		# NOTE sampling according to each tree.
		'colsample_bytree': trial.suggest_float('colsample_bytree', 0.445, 0.46),
		# NOTE sampling ratio for training data.
		'subsample': trial.suggest_float('subsample', 0.098, 1e-1),
		'eta': trial.suggest_float('eta', 0.4, 0.6),
		'max_depth': trial.suggest_int('max_depth', 3, 8),
		'max_leaves': trial.suggest_int('max_leaves', 3, 8),
		'min_child_weight': trial.suggest_int('min_child_weight', 0, 5),
		'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),
		# NOTE defines how selective algorithm is.
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
		# NOTE softmax for probability
		'objective': 'multi:softprob'
	}
	clf_xgb = xgb.XGBClassifier(**param)
	clf_xgb.set_params(eval_metric='merror', early_stopping_rounds=500) 
	clf_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
	preds = clf_xgb.predict(X_val)
	# weigthed_f1_score = f1_score(y_val, preds, average='weighted') # evaluation matric
	accuracy = accuracy_score(y_val, preds)
	return accuracy

'''
step 4: Optimization.
'''

n_train_iter = 100
sampler = optuna.samplers.TPESampler(seed=42, multivariate=True) 
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
study.optimize(objective, n_trials=n_train_iter, gc_after_trial=True)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

'''
step 5: Result Displays.
'''

# NOTE plot_optimization_histor: shows the scores from all trials as well as the best score so far at each point.
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
# NOTE plot_parallel_coordinate: interactively visualizes the hyperparameters and scores
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()
# NOTE plot_slice: shows the evolution of the search. You can see where in the hyperparameter space your search went and which parts of the space were explored more.
fig = optuna.visualization.plot_slice(study)
fig.show()
# NOTE plot_contour: plots parameter interactions on an interactive chart. You can choose which hyperparameters you would like to explore.
fig = optuna.visualization.plot_contour(study, params=['alpha',
                                                       'min_child_weight', 
                                                       'subsample', 
                                                       'learning_rate', 
                                                       'subsample'
                                                       ])
fig.show()
# NOTE plot parameter imprtances
fig = optuna.visualization.plot_param_importances(study)
fig.show()
#-----------------------------------------------------------------------------------------#
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#-----------------------------------------------------------------------------------------#
from sklearn.preprocessing import LabelEncoder
#-----------------------------------------------------------------------------------------#

main_folder = '../facies/'
lithofacies = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
if not os.path.exists(main_folder):
    os.makedirs(main_folder)
    print('creating folder: ', main_folder)
for facies in lithofacies:
    folder_path = os.path.join(main_folder, facies)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print('creating subfolder: ', facies)

well_logs = pd.read_csv('datasets/well_logs.csv')
well_logs.fillna(-999, inplace=True)
le = LabelEncoder()
well_logs['Facies'] = le.fit_transform(well_logs['Facies'])
drop_cols = ['Facies', 'Formation', 'Well Name', 'Depth'] 
X_train = well_logs.drop(drop_cols, axis=1) 
y_train = well_logs['Facies'] 

dummy = np.zeros(shape=(33, 33), dtype=np.float64)

for i in range(len(X_train)):
    label = lithofacies[y_train[i]]
    dummy[17, 16:23] = X_train.iloc[i, :].values
    # plt.imshow(dummy, cmap='rainbow')
    # plt.title(f'{label} - {i}')
    # plt.show()
    file_path = os.path.join(main_folder, label, f'{i:04d}.npy')
    np.save(file_path, dummy)
    print(file_path)
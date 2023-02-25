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

from PIL import Image
import matplotlib.cm as cm

for i in range(len(X_train)):
    label = lithofacies[y_train[i]]
    dummy[17, 16:23] = X_train.iloc[i, :].values
    # create a grayscale image
    img_arr = np.uint8(dummy * 255)
    img_arr = np.asarray(Image.fromarray(img_arr).resize((33, 33), resample=Image.BILINEAR))
    # apply a colormap to the grayscale image
    cmap = cm.get_cmap('jet')
    img_arr = (cmap(img_arr) * 255).astype(np.uint8)
    # save the image as a PNG file without title and axes
    file_path = os.path.join(main_folder, label, f'{i:04d}.png')
    Image.fromarray(img_arr).save(file_path, "PNG", optimize=True)
    print(file_path)
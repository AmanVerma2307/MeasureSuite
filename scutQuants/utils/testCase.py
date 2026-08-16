import numpy as np

X_train = np.load('./Datasets/SOLI/DGBQA-Seen/X_train_DGBQA-Seen_SOLI.npz',allow_pickle=True)['arr_0']
X_dev = np.load('./Datasets/SOLI/DGBQA-Seen/X_dev_DGBQA-Seen_SOLI.npz',allow_pickle=True)['arr_0']
y_train = np.load('./Datasets/SOLI/DGBQA-Seen/y_train_DGBQA-Seen_SOLI.npz',allow_pickle=True)['arr_0']
y_dev = np.load('./Datasets/SOLI/DGBQA-Seen/y_dev_DGBQA-Seen_SOLI.npz',allow_pickle=True)['arr_0']
y_train_id = np.load('./Datasets/SOLI/DGBQA-Seen/y_train_id_DGBQA-Seen_SOLI.npz',allow_pickle=True)['arr_0']
y_dev_id = np.load('./Datasets/SOLI/DGBQA-Seen/y_dev_id_DGBQA-Seen_SOLI.npz',allow_pickle=True)['arr_0']


print(X_train.shape, X_dev.shape)
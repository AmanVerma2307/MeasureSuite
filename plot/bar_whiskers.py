import sys
sys.path.insert(1,'./')
import numpy as np
import matplotlib.pyplot as plt
from utils.selector import get_val

####### Plot generation

###### Defining essentials
alpha = 2
beta = 0.75
kappa = 1
lambda_scale = 2
nu = 1

data_store = [] # List to store evaluation measures

e1_val = 100 - 16.45
e2_val = 100 - 23.36 

e1 = np.array([16.38,22.19,21.60,11.61,9.24,8.95,14.58,14.45,17.30,9.25,35.47])
e2 = np.array([21.12,26.42,32.30,20.34,18.18,17.33,19.81,24.45,25.70,11.52,39.81])

eer_values = (e1_val*e1+e2_val*e2)/(e1_val+e2_val)

G_total = 11
I_total = 26
y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_Tiny.npz')['arr_0']
y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_Tiny.npz')['arr_0']

embedding_list_full = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1pt5_Tiny.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-2pt5_Tiny.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1_Tiny.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1pt5_Tiny.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1-2pt5_Tiny.npz',
                       './Embeddings/MS_MF_1-1_Tiny.npz',
                       './Embeddings/MS_MF_1-1pt5_Tiny.npz',
                       './Embeddings/MS_MF_1-2pt5_Tiny.npz',
                       './Embeddings/MS_TAM_1-1_Tiny.npz',
                       './Embeddings/MS_TAM_1-1pt5_Tiny.npz',
                       './Embeddings/MS_TAM_1-2pt5_Tiny.npz']

##### Embedding generation
for embedding_path in embedding_list_full:

    embedding = np.load(embedding_path)['arr_0'] # Embedding path
    data_store.append(get_val(embedding,
                              y_dev,
                              y_dev_id,
                              eer_values,
                              G_total,
                              I_total,
                              'r',
                              'full',
                              quantifier='dgbqa'))

data_store = np.array(data_store) 
measure_count = int(data_store.shape[-1])
for measure_idx in range(measure_count): # Normalization
    data_store[:,measure_idx] = data_store[:,measure_idx]/np.linalg.norm(data_store[:,measure_idx])
data_store = np.transpose(data_store) # shape -> (measures,num_models)
data_store = list(data_store) # List conversion

##### Plotting
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,4))

labels = ['$\\mathcal{r}$',
          'R',
          '$\\psi$',
          '$C_{d}$',
          '$Ar(\Delta)$',
          '$Ar(\Delta)*\\bar{O}$',
          '$Ar(\Delta)*\\bar{\psi}$',
          '$\\bar{\psi}*\\bar{O}$',
          '$nAr^{*}(\Delta)$']

bplot = ax.boxplot(data_store,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks

colors = ['#29B6F6',
          '#4FC3F7',
          '#81D4FA',
          '#B3E5FC',
          '#E1F5FE',
          '#F8BBD0',
          '#F48FB1',
          '#F06292',
          '#EC307A']
colors = reversed(colors)

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xlabel('Evaluation measures\n(c) TinyRadar',fontsize=14)
ax.set_ylabel('Values',fontsize=14)
plt.show()
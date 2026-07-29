import sys
sys.path.insert(1,'./')

import numpy as np
import matplotlib.pyplot as plt
from RadarPlot import get_data, radar_factory
from utils.selector import select_model

####### Plotting
alpha = 2
beta = 0.75
kappa = 1
lambda_scale = 2
nu = 1

dataset_curr = 'Tiny'

if(dataset_curr == 'Soli'):
    y_dev_path = './Embeddings/y_dev_DeltaDistance_SOLI.npz'
    y_dev_id_path = './Embeddings/y_dev_id_DeltaDistance_SOLI.npz'
    G_total = 11
    I_total = 10
    eer_values = np.array([15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63])

if(dataset_curr == 'HandLogin'):
    y_dev_path = './Embeddings/y_dev_DGBQA_Seen_HandLogin.npz'
    y_dev_id_path = './Embeddings/y_dev_id_DGBQA_Seen_HandLogin.npz'
    G_total = 4
    I_total = 16
    eer_values = np.array([0.44,1.29,4.89,1.05])

if(dataset_curr == 'Tiny'):
    y_dev_path = './Embeddings/y_dev_DGBQA_Seen_Tiny.npz'
    y_dev_id_path = './Embeddings/y_dev_id_DGBQA_Seen_Tiny.npz'
    G_total = 11
    I_total = 26

    e1_val = 100 - 16.45
    e2_val = 100 - 23.36 
    e1 = np.array([16.38,22.19,21.60,11.61,9.24,8.95,14.58,14.45,17.30,9.25,35.47])
    e2 = np.array([21.12,26.42,32.30,20.34,18.18,17.33,19.81,24.45,25.70,11.52,39.81])
    eer_values = (e1_val*e1+e2_val*e2)/(e1_val+e2_val)
    eer_values = eer_values

##### Defining essentials
embedding_list_soli = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1-pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-1_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-1pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_pt5-pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_pt5-1_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_pt5-1pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_1-pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_1-1_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_1-1pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-1_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-1pt5_SOLI.npz',
                        './Embeddings/MS_TPN_pt5-pt5_SOLI.npz',
                        './Embeddings/MS_TPN_pt5-1_SOLI.npz',
                        './Embeddings/MS_TPN_pt5-1pt5_SOLI.npz',
                        './Embeddings/MS_TPN_1-pt5_SOLI.npz',
                        './Embeddings/MS_TPN_1-1_SOLI.npz',
                        './Embeddings/MS_TPN_1-1pt5_SOLI.npz',
                        './Embeddings/MS_TPN_1pt5-pt5_SOLI.npz',
                        './Embeddings/MS_TPN_1pt5-1_SOLI.npz',
                        './Embeddings/MS_TPN_1pt5-1pt5_SOLI.npz',
                        './Embeddings/MS_TAM_pt5-pt5_SOLI.npz',
                        './Embeddings/MS_TAM_1-pt5_SOLI.npz',
                        './Embeddings/MS_TAM_1-1_SOLI.npz',
                        './Embeddings/MS_MViT_pt5-pt5_SOLI.npz',
                        './Embeddings/MS_MViT_pt5-1_SOLI.npz',
                        './Embeddings/MS_MViT_pt5-1pt5_SOLI.npz',
                        './Embeddings/MS_MViT_1-pt5_SOLI.npz',
                        './Embeddings/MS_MViT_1-1_SOLI.npz',
                        './Embeddings/MS_MViT_1-1pt5_SOLI.npz',
                        './Embeddings/MS_MViT_1pt5-pt5_SOLI.npz',
                        './Embeddings/MS_MViT_1pt5-1_SOLI.npz',
                        './Embeddings/MS_MViT_1pt5-1pt5_SOLI.npz']
dataset_list_soli = ['Soli']*39

embedding_list_hl = ['./Embeddings/MS_ViViT_pt5-pt5_HandLogin.npz',
                       './Embeddings/MS_ViViT_pt5-1_HandLogin.npz',
                       './Embeddings/MS_ViViT_pt5-1pt5_HandLogin.npz',
                       './Embeddings/MS_ViViT_pt5-2pt5_HandLogin.npz',
                       './Embeddings/MS_ViViT_1-pt5_HandLogin.npz',
                       './Embeddings/MS_ViViT_1-1_HandLogin.npz',
                       './Embeddings/MS_ViViT_1-1pt5_HandLogin.npz',
                       './Embeddings/MS_ViViT_1-2pt5_HandLogin.npz',
                       './Embeddings/MS_ViViT_1pt5-pt5_HandLogin.npz',
                       './Embeddings/MS_ViViT_1pt5-1_HandLogin.npz',
                       './Embeddings/MS_ViViT_1pt5-1pt5_HandLogin.npz',
                       './Embeddings/MS_ViViT_1pt5-2pt5_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1-pt5_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1-1_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1-1pt5_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1-2pt5_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1pt5-pt5_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1pt5-1_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1pt5-1pt5_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_pt5-1_HandLogin.npz',
                        './Embeddings/MS_TPN_pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_1-pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_1-1pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_1-2pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_1pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_1pt5-1_HandLogin.npz',
                        './Embeddings/MS_TPN_1pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_1pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_pt5-1_HandLogin.npz',
                        './Embeddings/MS_TAM_pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_1-pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_1-1_HandLogin.npz',
                        './Embeddings/MS_TAM_1-1pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_1-2pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_1pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_1pt5-1_HandLogin.npz',
                        './Embeddings/MS_TAM_1pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_1pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_pt5-1_HandLogin.npz',
                        './Embeddings/MS_MViT_pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_1-pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_1-1_HandLogin.npz',
                        './Embeddings/MS_MViT_1-1pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_1-2pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_1pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_1pt5-1_HandLogin.npz',
                        './Embeddings/MS_MViT_1pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_1pt5-2pt5_HandLogin.npz']
dataset_list_hl = ['HandLogin']*55

embedding_list_tiny = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz',
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
dataset_list_tiny = ['Tiny']*12

model_list = []

for measure in ['r','R','psi','Cd','Ar','ArCd','Ar_psi','Cd_psi','Ar*']:
    model_list.append(select_model(embedding_list_tiny,
                                   dataset_list_tiny,
                                   measure
                                   ))
print(model_list)

#model_list = ['./Embeddings/MS_TPN_1pt5-1_SOLI.npz', 
#              './Embeddings/MS_TPN_1pt5-1_SOLI.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1_SOLI.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-1_SOLI.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-MF_1-1_SOLI.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-MF_1-1pt5_SOLI.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-pt5_SOLI.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1pt5_SOLI.npz', 
#              './Embeddings/MS_MViT_pt5-1_SOLI.npz',
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz',
#              './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-pt5_SOLI.npz',
#              './Embeddings/MS_MViT_pt5-1_SOLI.npz']

#model_list = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_1-2pt5_Tiny.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_1-2pt5_Tiny.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_1-2pt5_Tiny.npz', 
#              './Embeddings/MS_MF_1-1_Tiny.npz', 
#              './Embeddings/MS_MF_1-1pt5_Tiny.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-2pt5_Tiny.npz', 
#              './Embeddings/DGBQA_CGID_Res3D-ViViT_1-2pt5_Tiny.npz',
#              './Embeddings/MS_MF_1-1pt5_Tiny.npz']
              
data = get_data(model_list,
                y_dev_path,
                y_dev_id_path,
                eer_values,
                G_total,
                I_total,
                alpha=2,
                beta=0.75,
                lambda_scale=2,
                kappa=1,
                nu=1,
                num_entries=9)
N = 4
theta = radar_factory(N, frame='polygon')
colors = ['tab:blue',
          'tab:orange',
          'tab:green',
          'tab:purple',
          'tab:gray',
          'tab:olive',
          'indigo',
          'tab:cyan',  
          'tab:red']
spoke_labels = data.pop(0)
#print(data)

fig, ax = plt.subplots(figsize=(9,8), nrows=1, ncols=1,
                        subplot_kw=dict(projection='radar'))

for d, color in zip(data, colors):
    ax.plot(theta,d,color=color,marker='o',markersize=10)
    ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
ax.set_varlabels(spoke_labels)
ax.set_xlabel('(c) TinyRadar',fontsize=14)

labels = ['$\\mathcal{r}$: ViViT(0.5,1.5)',
          '$\\mathcal{R}$: ViViT(1.0,2.5)',
          '$\\psi$: TAM(1.0,1.0)',
          '$C_{d}$: ViViT(0.5,2.5)',
          '$A_r(\Delta)$: MF(1.0,1.5)',
          '$A_r(\Delta)*\\bar{O}$: MF(1.0,1.5)',
          '$A_r(\Delta)*\\bar{\psi}$: MF(1.0,1.5)',
          '$\\bar{\psi}*\\bar{O}$: MF(1.0,1.5)',
          '$nA_r^{*}(\Delta)$: MF(1.0,1.5)']

legend = ax.legend(labels,
                   loc=(0.75,0.75),
                   labelspacing=0.1, 
                   fontsize='14')

plt.show()


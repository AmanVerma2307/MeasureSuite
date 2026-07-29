import sys
sys.path.insert(1,'./')

import numpy as np
import matplotlib.pyplot as plt
from src.DGBQA_Score import gbqa_delta_dist_compute

####### Rank deviation curves
dataset_curr = 'Soli'
embedding = np.load('./Embeddings/MS_MViT_pt5-1_SOLI.npz')['arr_0']
labels = ['Pinch index','Palm tilt','Finger Slider','Pinch pinky','Slow Swipe','Fast Swipe','Push','Pull','Finger rub','Circle','Palm hold']

if(dataset_curr == 'Soli'):
    y_dev = np.load('./Embeddings/y_dev_DeltaDistance_SOLI.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DeltaDistance_SOLI.npz')['arr_0']
    G_total = 11
    I_total = 10
    eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]

if(dataset_curr == 'HandLogin'):
    y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_HandLogin.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_HandLogin.npz')['arr_0']
    G_total = 4
    I_total = 16
    eer_values = [0.44,1.29,4.89,1.05]

if(dataset_curr == 'Tiny'):
    y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_Tiny.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_Tiny.npz')['arr_0']
    G_total = 11
    I_total = 26

    e1_val = 100 - 16.45
    e2_val = 100 - 23.36 
    e1 = np.array([16.38,22.19,21.60,11.61,9.24,8.95,14.58,14.45,17.30,9.25,35.47])
    e2 = np.array([21.12,26.42,32.30,20.34,18.18,17.33,19.81,24.45,25.70,11.52,39.81])
    eer_values = (e1_val*e1+e2_val*e2)/(e1_val+e2_val)
    eer_values = list(eer_values)

##### DGBQA score computation
dgbqa_score = [] # DGBQA Score
for g_id in range(G_total):
    dgbqa_score_curr, _, _, _ = gbqa_delta_dist_compute(embedding,g_id,I_total,y_dev,y_dev_id)
    dgbqa_score.append(dgbqa_score_curr)

dgbqa_score = np.array(dgbqa_score) # Array Formation
dgbqa_score = (dgbqa_score - np.mean(dgbqa_score))/np.std(dgbqa_score) # Mean Normalization
dgbqa_score = dgbqa_score/np.linalg.norm(dgbqa_score) # L2-Normalization

##### Ground truth equal error rates
e_prime = 100 - np.array(eer_values)
e_prime = (e_prime - np.mean(e_prime))/np.std(e_prime)
e_prime = e_prime/np.linalg.norm(e_prime)

##### Plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
x_axes = np.arange(start=0,stop=22,step=2)
ax.bar(x_axes,e_prime,label='DGBQA score',color='gold')
ax.bar(x_axes+0.8,dgbqa_score,label='Ground truth score',color='tab:red')
#ax.set_xlabel('(a) $r:$ Res3D-ViViT (1.5,0.5)',fontsize=14)
ax.set_xticks(x_axes+0.5)
ax.set_xticklabels(labels=labels,fontsize=5,rotation=30)
#ax.set_yticks(fontsize=8)
ax.tick_params(bottom=True,left=True)
ax.legend(frameon=True,fontsize=8)
plt.show()
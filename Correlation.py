import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from selector import get_val, get_params, make_df

###### Parameter generation

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

#measure_val_soli = get_params(embedding_list_soli,
#                            dataset_list_soli,
#                            'full')
#df_soli = make_df(np.array(measure_val_soli))


#measure_val_hl = get_params(embedding_list_hl,
#                            dataset_list_hl,
#                            'full')
#df_hl = make_df(np.array(measure_val_hl))

measure_val_tiny = get_params(embedding_list_tiny,dataset_list_tiny,'full')
df_tiny = make_df(np.array(measure_val_tiny))

#measure_val_tiny = get_params(embedding_list_tiny,
#                                       dataset_list_tiny,
#                                       'full')
#measure_val = np.array(measure_val_soli+measure_val_hl+measure_val_tiny)
#measure_val = np.load('./measure_val.npz',allow_pickle=True)['arr_0']
#df = make_df(np.array(measure_val))
#df = make_df(measure_val[:39])
#df['dataset'] = dataset_list_soli
#np.savez_compressed('./measure_val.npz',measure_val)

labels = ['$\\mathcal{r}$',
          'R',
          '$\\psi$',
          '$C_{d}$',
          '$Ar(\Delta)$',
          '$Ar(\Delta)*\\bar{O}$',
          '$Ar(\Delta)*\\bar{\psi}$',
          '$\\bar{\psi}*\\bar{O}$',
          '$nAr^{*}(\Delta)$']


###### Plot generation
rocket = sns.color_palette("rocket")
sns.jointplot(x="rpp",y="Ar_star",data=df_tiny,kind="reg",color=rocket[5])
plt.ylabel('$nA_r^{*}(\Delta)$',fontsize=14)
plt.xlabel('RPP',fontsize=14)
plt.show()

#fig, ((ax1,ax2,ax3,ax4,ax5,ax6,ax7),
#      (ax8,ax9,ax10,ax11,ax12,ax13,ax14),
#      (ax15,ax16,ax17,ax18,ax19,ax20,ax21)) = plt.subplots(nrows=3,
#                                      ncols=7,
#                                      figsize=(12,8))
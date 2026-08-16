import sys
sys.path.insert(1,'./')

import numpy as np
import matplotlib.pyplot as plt
from utils.hyperparameter import get_nar

####### Testing
#embedding_list = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz',
#                  './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-pt5_SOLI.npz',
#                  './Embeddings/MS_TPN_pt5-pt5_SOLI.npz',
#                  './Embeddings/MS_TAM_1-pt5_SOLI.npz',
#                  './Embeddings/MS_MViT_pt5-1_SOLI.npz']
#y_dev_path = './Embeddings/y_dev_DeltaDistance_SOLI.npz'
#y_dev_id_path = './Embeddings/y_dev_id_DeltaDistance_SOLI.npz'
#hyp_val = [0.25,0.50,0.75,1.00,2.00,4.00]
#eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]
#nar_values = []

#for hyp_val_curr in hyp_val:
#    nar_values.append(get_nar(embedding_list,
#                  y_dev_path,
#                  y_dev_id_path,
#                  eer_values,
#                  11,
#                  10,
#                  alpha=2,
#                  beta=0.75,
#                  lambda_scale=2,
#                  kappa=hyp_val_curr,
#                  nu=1))
    
#nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

#plt.plot(hyp_val,nar_values[:,0],label='Res3D-ViViT',linewidth=3,marker='o',markersize=8)
#plt.plot(hyp_val,nar_values[:,1],label='Res3D-MF',linewidth=3,marker='o',markersize=8)
#plt.plot(hyp_val,nar_values[:,2],label='Res3D-TPN',linewidth=3,marker='o',markersize=8)
#plt.plot(hyp_val,nar_values[:,3],label='Res3D-TAM',linewidth=3,marker='o',markersize=8)
##plt.plot(hyp_val,nar_values[:,4],label='Res3D-MF',linewidth=3,marker='o',markersize=8)
#plt.legend()
#plt.show()

####### Curve Plotting
hyp_val = [0.25,0.50,0.75,1.00,2.00,4.00]
fig, ((ax1,ax2,ax3,ax4),
      (ax5,ax6,ax7,ax8),
      (ax9,ax10,ax11,ax12)) = plt.subplots(nrows=3,
                                      ncols=4,
                                      figsize=(12,8))

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:

    if(ax == ax1 or ax == ax2 or ax == ax3 or ax == ax4): 

        embedding_list = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz',
                  './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-pt5_SOLI.npz',
                  './Embeddings/MS_TPN_pt5-pt5_SOLI.npz',
                  './Embeddings/MS_TAM_1-pt5_SOLI.npz',
                  './Embeddings/MS_MViT_pt5-1_SOLI.npz']
        
        y_dev_path = './Embeddings/y_dev_DeltaDistance_SOLI.npz'
        y_dev_id_path = './Embeddings/y_dev_id_DeltaDistance_SOLI.npz'
        eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]

        if(ax == ax1):          
            nar_values = []
            for hyp_val_curr in hyp_val:
                nar_values.append(get_nar(embedding_list,
                              y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              11,
                              10,
                              alpha=2,
                              beta=0.75,
                              lambda_scale=2,
                              kappa=hyp_val_curr,
                              nu=1))    
            nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

            ax.plot(hyp_val,nar_values[:,0],label='ViViT',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,1],label='MF',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,2],label='TPN',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,3],label='TAM',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,4],label='MViT',linewidth=3,marker='o',markersize=8)
            ax.legend(frameon=True,fontsize=8)
            ax.set_xlabel("$\\kappa$\n(a) Soli: $\\kappa$",fontsize=10)            
            ax.set_ylabel('$nA_r^{*}(\Delta)$',fontsize=12)

        if(ax == ax2):          
            nar_values = []
            for hyp_val_curr in hyp_val:
                nar_values.append(get_nar(embedding_list,
                              y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              11,
                              10,
                              alpha=2,
                              beta=hyp_val_curr,
                              lambda_scale=2,
                              kappa=1,
                              nu=1))    
            nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

            ax.plot(hyp_val,nar_values[:,0],label='ViViT',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,1],label='MF',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,2],label='TPN',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,3],label='TAM',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,4],label='MViT',linewidth=3,marker='o',markersize=8)
            ax.legend(frameon=True,fontsize=8)
            ax.set_xlabel("$\\beta$\n(b) Soli: $\\beta$",fontsize=10)     
            ax.set_ylabel('$nA_r^{*}(\Delta)$',fontsize=12)

        if(ax == ax3):          
            nar_values = []
            for hyp_val_curr in hyp_val:
                nar_values.append(get_nar(embedding_list,
                              y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              11,
                              10,
                              alpha=2,
                              beta=0.75,
                              lambda_scale=hyp_val_curr,
                              kappa=1,
                              nu=1))    
            nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

            ax.plot(hyp_val,nar_values[:,0],label='ViViT',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,1],label='MF',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,2],label='TPN',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,3],label='TAM',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,4],label='MViT',linewidth=3,marker='o',markersize=8)
            ax.legend(frameon=True,fontsize=8)
            ax.set_xlabel("$\\lambda$\n(c) Soli: $\\lambda$",fontsize=10)     
            ax.set_ylabel('$nA_r^{*}(\Delta)$',fontsize=12)

        if(ax == ax4):          
            nar_values = []
            for hyp_val_curr in hyp_val:
                nar_values.append(get_nar(embedding_list,
                              y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              11,
                              10,
                              alpha=2,
                              beta=0.75,
                              lambda_scale=2,
                              kappa=1,
                              nu=hyp_val_curr))    
            nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

            ax.plot(hyp_val,nar_values[:,0],label='ViViT',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,1],label='MF',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,2],label='TPN',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,3],label='TAM',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,4],label='MViT',linewidth=3,marker='o',markersize=8)
            ax.legend(frameon=True,fontsize=8)
            ax.set_xlabel("$\\nu$\n(d) Soli: $\\nu$",fontsize=10)     
            ax.set_ylabel('$nA_r^{*}(\Delta)$',fontsize=12)

    if(ax == ax5 or ax == ax6 or ax == ax7 or ax == ax8): 

        embedding_list = ['./Embeddings/MS_ViViT_pt5-pt5_HandLogin.npz',
                  './Embeddings/Test/DGBQA_CGID_Res3D-MF_1-pt5_HandLogin.npz',
                  './Embeddings/MS_TPN_1pt5-1_HandLogin.npz',
                  './Embeddings/MS_TAM_1-2pt5_HandLogin.npz',
                  './Embeddings/MS_MViT_1-1pt5_HandLogin.npz']
        
        y_dev_path = './Embeddings/y_dev_DGBQA_Seen_HandLogin.npz'
        y_dev_id_path = './Embeddings/y_dev_id_DGBQA_Seen_HandLogin.npz'
        eer_values = [0.44,1.29,4.89,1.05]

        if(ax == ax5):
            nar_values = []
            for hyp_val_curr in hyp_val:
                nar_values.append(get_nar(embedding_list,
                              y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              4,
                              16,
                              alpha=2,
                              beta=0.75,
                              lambda_scale=2,
                              kappa=hyp_val_curr,
                              nu=1))    
            nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

            ax.plot(hyp_val,nar_values[:,0],label='ViViT',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,1],label='MF',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,2],label='TPN',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,3],label='TAM',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,4],label='MViT',linewidth=3,marker='o',markersize=8)
            ax.legend(frameon=True,fontsize=8)
            ax.set_xlabel("$\\kappa$\n(e) HandLogin: $\\kappa$",fontsize=10)     
            ax.set_ylabel('$nA_r^{*}(\Delta)$',fontsize=12)

        if(ax == ax6):
            nar_values = []
            for hyp_val_curr in hyp_val:
                nar_values.append(get_nar(embedding_list,
                              y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              4,
                              16,
                              alpha=2,
                              beta=hyp_val_curr,
                              lambda_scale=2,
                              kappa=1,
                              nu=1))    
            nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

            ax.plot(hyp_val,nar_values[:,0],label='ViViT',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,1],label='MF',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,2],label='TPN',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,3],label='TAM',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,4],label='MViT',linewidth=3,marker='o',markersize=8)
            ax.legend(frameon=True,fontsize=8)
            ax.set_xlabel("$\\beta$\n(f) HandLogin: $\\beta$",fontsize=10)     
            ax.set_ylabel('$nA_r^{*}(\Delta)$',fontsize=12)
        
        if(ax == ax7):
            nar_values = []
            for hyp_val_curr in hyp_val:
                nar_values.append(get_nar(embedding_list,
                              y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              4,
                              16,
                              alpha=2,
                              beta=0.75,
                              lambda_scale=hyp_val_curr,
                              kappa=1,
                              nu=1))    
            nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

            ax.plot(hyp_val,nar_values[:,0],label='ViViT',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,1],label='MF',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,2],label='TPN',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,3],label='TAM',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,4],label='MViT',linewidth=3,marker='o',markersize=8)
            ax.legend(frameon=True,fontsize=8)
            ax.set_xlabel("$\\lambda$\n(g) HandLogin: $\\lambda$",fontsize=10)     
            ax.set_ylabel('$nA_r^{*}(\Delta)$',fontsize=12)

        if(ax == ax8):
            nar_values = []
            for hyp_val_curr in hyp_val:
                nar_values.append(get_nar(embedding_list,
                              y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              4,
                              16,
                              alpha=2,
                              beta=0.75,
                              lambda_scale=2,
                              kappa=1,
                              nu=hyp_val_curr))    
            nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

            ax.plot(hyp_val,nar_values[:,0],label='ViViT',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,1],label='MF',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,2],label='TPN',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,3],label='TAM',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,4],label='MViT',linewidth=3,marker='o',markersize=8)
            ax.legend(frameon=True,fontsize=8)
            ax.set_xlabel("$\\nu$\n(h) HandLogin: $\\nu$",fontsize=10)     
            ax.set_ylabel('$nA_r^{*}(\Delta)$',fontsize=12)

    if(ax == ax9 or ax == ax10 or ax == ax11 or ax == ax12): 

        embedding_list = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_1-1_Tiny.npz',
                  './Embeddings/MS_MF_1-1pt5_Tiny.npz',
                  './Embeddings/MS_TAM_1-2pt5_Tiny.npz']
        
        y_dev_path = './Embeddings/y_dev_DGBQA_Seen_Tiny.npz'
        y_dev_id_path = './Embeddings/y_dev_id_DGBQA_Seen_Tiny.npz'
        
        e1_val = 100 - 16.45
        e2_val = 100 - 23.36 

        e1 = np.array([16.38,22.19,21.60,11.61,9.24,8.95,14.58,14.45,17.30,9.25,35.47])
        e2 = np.array([21.12,26.42,32.30,20.34,18.18,17.33,19.81,24.45,25.70,11.52,39.81])

        eer_values = (e1_val*e1+e2_val*e2)/(e1_val+e2_val)

        if(ax == ax9):
            nar_values = []
            for hyp_val_curr in hyp_val:
                nar_values.append(get_nar(embedding_list,
                              y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              11,
                              26,
                              alpha=2,
                              beta=0.75,
                              lambda_scale=2,
                              kappa=hyp_val_curr,
                              nu=1))    
            nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

            ax.plot(hyp_val,nar_values[:,0],label='ViViT',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,1],label='MF',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,2],label='TAM',linewidth=3,marker='o',markersize=8)
            ax.legend(frameon=True,fontsize=8)
            ax.set_xlabel("$\\kappa$\n(i) TinyRadar: $\\kappa$",fontsize=10)     
            ax.set_ylabel('$nA_r^{*}(\Delta)$',fontsize=12)

        if(ax == ax10):
            nar_values = []
            for hyp_val_curr in hyp_val:
                nar_values.append(get_nar(embedding_list,
                              y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              11,
                              26,
                              alpha=2,
                              beta=hyp_val_curr,
                              lambda_scale=2,
                              kappa=1,
                              nu=1))    
            nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

            ax.plot(hyp_val,nar_values[:,0],label='ViViT',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,1],label='MF',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,2],label='TAM',linewidth=3,marker='o',markersize=8)
            ax.legend(frameon=True,fontsize=8)
            ax.set_xlabel("$\\beta$\n(j) TinyRadar: $\\beta$",fontsize=10)     
            ax.set_ylabel('$nA_r^{*}(\Delta)$',fontsize=12)

        if(ax == ax11):
            nar_values = []
            for hyp_val_curr in hyp_val:
                nar_values.append(get_nar(embedding_list,
                              y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              11,
                              26,
                              alpha=2,
                              beta=0.75,
                              lambda_scale=hyp_val_curr,
                              kappa=1,
                              nu=1))    
            nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

            ax.plot(hyp_val,nar_values[:,0],label='ViViT',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,1],label='MF',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,2],label='TAM',linewidth=3,marker='o',markersize=8)
            ax.legend(frameon=True,fontsize=8)
            ax.set_xlabel("$\\lambda$\n(k) TinyRadar: $\\lambda$",fontsize=10)     
            ax.set_ylabel('$nA_r^{*}(\Delta)$',fontsize=12)

        if(ax == ax12):
            nar_values = []
            for hyp_val_curr in hyp_val:
                nar_values.append(get_nar(embedding_list,
                              y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              11,
                              26,
                              alpha=2,
                              beta=0.75,
                              lambda_scale=2,
                              kappa=1,
                              nu=hyp_val_curr))    
            nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

            ax.plot(hyp_val,nar_values[:,0],label='ViViT',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,1],label='MF',linewidth=3,marker='o',markersize=8)
            ax.plot(hyp_val,nar_values[:,2],label='TAM',linewidth=3,marker='o',markersize=8)
            ax.legend(frameon=True,fontsize=8)
            ax.set_xlabel("$\\nu$\n(l) TinyRadar: $\\nu$",fontsize=10)     
            ax.set_ylabel('$nA_r^{*}(\Delta)$',fontsize=12)

plt.show()

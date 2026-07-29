import sys
sys.path.insert(1,'./')

import numpy as np
import matplotlib.pyplot as plt
from utils.hyperparameter import compute_acceptance

####### Surface Plot
def get_nar(embedding_list,
                 y_dev_path, 
                 y_dev_id_path, 
                 eer_values, 
                 G_total,
                 I_total,
                 alpha,
                 beta,
                 lambda_scale,
                 kappa,
                 nu):
    
    """
    Function to plot surface plot wrt lambda values 
    """

    y_dev = np.load(y_dev_path)['arr_0']
    y_dev_id = np.load(y_dev_id_path)['arr_0']
    nar_value = []

    for embedding_path in embedding_list:
        embedding = np.load(embedding_path)['arr_0']
        nar_value.append(compute_acceptance(embedding,
                                            y_dev,
                                            y_dev_id,
                                            eer_values,
                                            G_total,
                                            I_total,
                                            alpha,
                                            beta,
                                            lambda_scale,
                                            kappa,
                                            nu))
    return np.array(nar_value)
        
###### Surface Plot
def plot_surface(nar_val):

    #print(nar_val)
    ax = plt.figure().add_subplot(projection='3d')
    
    X = Y = np.arange(0.5,2.0,0.5)
    X, Y = np.meshgrid(X,Y)
    Z = np.array(nar_val).reshape(X.shape)
    #print(X,Y,Z)
    ax.plot_wireframe(Y,X,Z,
                    lw=2.0)
    ax.plot_surface(Y,X,Z,alpha=0.3)
    ax.contourf(Y, X, Z, zdir='z', offset=0.05, cmap='Blues')
    
    X = np.array([0.5,0.5,0.5,1.0,1.0,1.0,1.5,1.5,1.5])
    Y = np.array([0.5,1.0,1.5,0.5,1.0,1.5,0.5,1.0,1.5])
    Z = np.array(nar_val)

    ax.scatter(X,Y,Z,
               s=200,
               color='black',
               edgecolors='k',
               marker='h')
    #ax.stem(X,Y,Z)

    ax.set_zlabel('$nAr^{*}(\Delta)$',fontsize=12)
    ax.set_xlabel('$\lambda_{ID}$',fontsize=12) 
    ax.set_ylabel('$\lambda_{ICGD}$',fontsize=12)

    ax.view_init(elev=17., azim=30) 
    ax.set_zlim(0.05,0.65)
    plt.show()

###### Defining essentials
embedding_list = ['./Embeddings/MS_MViT_pt5-pt5_SOLI.npz',
                  './Embeddings/MS_MViT_pt5-1_SOLI.npz',
                  './Embeddings/MS_MViT_pt5-1pt5_SOLI.npz',
                  './Embeddings/MS_MViT_1-pt5_SOLI.npz',
                  './Embeddings/MS_MViT_1-1_SOLI.npz',
                  './Embeddings/MS_MViT_1-1pt5_SOLI.npz',
                  './Embeddings/MS_MViT_1pt5-pt5_SOLI.npz',
                  './Embeddings/MS_MViT_1pt5-1_SOLI.npz',
                  './Embeddings/MS_MViT_1pt5-1pt5_SOLI.npz']
y_dev_path = './Embeddings/y_dev_DeltaDistance_SOLI.npz'
y_dev_id_path = './Embeddings/y_dev_id_DeltaDistance_SOLI.npz'
eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]
G_total=11
I_total=10
X = Y = np.arange(0.5,2.0,0.5)
X, Y = np.meshgrid(X,Y)
X_scatter = np.array([0.5,0.5,0.5,1.0,1.0,1.0,1.5,1.5,1.5])
Y_scatter = np.array([0.5,1.0,1.5,0.5,1.0,1.5,0.5,1.0,1.5])
#Z = np.array(nar_val).reshape(X.shape)
#plot_surface(get_nar(embedding_list,
#                     y_dev_path,
#                     y_dev_id_path,
#                     eer_values,
#                     11,
#                     10,
#                     alpha=2,
#                     beta=0.75,
#                     lambda_scale=2,
#                     kappa=1,
#                    nu=1))

###### Plotting
fig, (ax11, ax12, ax13, ax14, ax15, ax16) = plt.subplots(nrows=1, ncols=6, figsize=(18,6), subplot_kw=dict(projection='3d'))

for ax in [ax11, ax12, ax13, ax14, ax15, ax16]:
    
    ##### Kappa
    #### K=1/4
    if(ax==ax11):
        nar_val = get_nar(embedding_list,y_dev_path,y_dev_id_path,
                     eer_values,
                     11,10,
                     alpha=2,
                     beta=0.75,
                     lambda_scale=2,
                     kappa=1,
                    nu=0.25)
        Z = np.array(nar_val).reshape(X.shape)
        ax.plot_wireframe(Y,X,Z,
                    lw=2.0)
        ax.plot_surface(Y,X,Z,alpha=0.3)
        ax.contourf(Y, X, Z, zdir='z', offset=0.05, cmap='Blues')
        Z_scatter = np.array(nar_val)
        ax.scatter(X_scatter,Y_scatter,Z_scatter,
                s=100,
                color='black',
                edgecolors='k',
                marker='h')
        ax.set_zlabel('$nAr^{*}(\Delta)$',fontsize=12)
        ax.set_xlabel('$\lambda_{ID}$',fontsize=12) 
        ax.set_ylabel('$\lambda_{ICGD}$',fontsize=12)
        ax.set_title('(d.i) $\\nu=0.25$',fontsize=14)
        ax.view_init(elev=17., azim=30) 
        ax.set_zlim(0.05,0.65)

    #### K=1/2
    if(ax==ax12):
        nar_val = get_nar(embedding_list,y_dev_path,y_dev_id_path,
                     eer_values,
                     11,10,
                     alpha=2,
                     beta=0.75,
                     lambda_scale=2,
                     kappa=1,
                    nu=0.50)
        Z = np.array(nar_val).reshape(X.shape)
        ax.plot_wireframe(Y,X,Z,
                    lw=2.0)
        ax.plot_surface(Y,X,Z,alpha=0.3)
        ax.contourf(Y, X, Z, zdir='z', offset=0.05, cmap='Blues')
        Z_scatter = np.array(nar_val)
        ax.scatter(X_scatter,Y_scatter,Z_scatter,
                s=100,
                color='black',
                edgecolors='k',
                marker='h')
        ax.set_zlabel('$nAr^{*}(\Delta)$',fontsize=12)
        ax.set_xlabel('$\lambda_{ID}$',fontsize=12) 
        ax.set_ylabel('$\lambda_{ICGD}$',fontsize=12)
        ax.set_title('(d.ii) $\\nu=0.50$',fontsize=14)
        ax.view_init(elev=17., azim=30) 
        ax.set_zlim(0.05,0.65)

    #### K=3/4
    if(ax==ax13):
        nar_val = get_nar(embedding_list,y_dev_path,y_dev_id_path,
                     eer_values,
                     11,10,
                     alpha=2,
                     beta=0.75,
                     lambda_scale=2,
                     kappa=1,
                    nu=0.75)
        Z = np.array(nar_val).reshape(X.shape)
        ax.plot_wireframe(Y,X,Z,
                    lw=2.0)
        ax.plot_surface(Y,X,Z,alpha=0.3)
        ax.contourf(Y, X, Z, zdir='z', offset=0.05, cmap='Blues')
        Z_scatter = np.array(nar_val)
        ax.scatter(X_scatter,Y_scatter,Z_scatter,
                s=100,
                color='black',
                edgecolors='k',
                marker='h')
        ax.set_zlabel('$nAr^{*}(\Delta)$',fontsize=12)
        ax.set_xlabel('$\lambda_{ID}$',fontsize=12) 
        ax.set_ylabel('$\lambda_{ICGD}$',fontsize=12)
        ax.set_title('(d.iii) $\\nu=0.75$',fontsize=14)
        ax.view_init(elev=17., azim=30) 
        ax.set_zlim(0.05,0.65)

    #### K=1
    if(ax==ax14):
        nar_val = get_nar(embedding_list,y_dev_path,y_dev_id_path,
                     eer_values,
                     11,10,
                     alpha=2,
                     beta=0.75,
                     lambda_scale=2,
                     kappa=1,
                    nu=1.00)
        Z = np.array(nar_val).reshape(X.shape)
        ax.plot_wireframe(Y,X,Z,
                    lw=2.0)
        ax.plot_surface(Y,X,Z,alpha=0.3)
        ax.contourf(Y, X, Z, zdir='z', offset=0.05, cmap='Blues')
        Z_scatter = np.array(nar_val)
        ax.scatter(X_scatter,Y_scatter,Z_scatter,
                s=100,
                color='black',
                edgecolors='k',
                marker='h')
        ax.set_zlabel('$nAr^{*}(\Delta)$',fontsize=12)
        ax.set_xlabel('$\lambda_{ID}$',fontsize=12) 
        ax.set_ylabel('$\lambda_{ICGD}$',fontsize=12)
        ax.set_title('(d.iv) $\\nu=1.00$',fontsize=14)
        ax.view_init(elev=17., azim=30) 
        ax.set_zlim(0.05,0.65)

    #### K=2
    if(ax==ax15):
        nar_val = get_nar(embedding_list,y_dev_path,y_dev_id_path,
                     eer_values,
                     11,10,
                     alpha=2,
                     beta=0.75,
                     lambda_scale=2,
                     kappa=1,
                    nu=2.00)
        Z = np.array(nar_val).reshape(X.shape)
        ax.plot_wireframe(Y,X,Z,
                    lw=2.0)
        ax.plot_surface(Y,X,Z,alpha=0.3)
        ax.contourf(Y, X, Z, zdir='z', offset=0.05, cmap='Blues')
        Z_scatter = np.array(nar_val)
        ax.scatter(X_scatter,Y_scatter,Z_scatter,
                s=100,
                color='black',
                edgecolors='k',
                marker='h')
        ax.set_zlabel('$nAr^{*}(\Delta)$',fontsize=12)
        ax.set_xlabel('$\lambda_{ID}$',fontsize=12) 
        ax.set_ylabel('$\lambda_{ICGD}$',fontsize=12)
        ax.set_title('(d.v) $\\nu=2.00$',fontsize=14)
        ax.view_init(elev=17., azim=30) 
        ax.set_zlim(0.05,0.65)

    #### K=4
    if(ax==ax16):
        nar_val = get_nar(embedding_list,y_dev_path,y_dev_id_path,
                     eer_values,
                     11,10,
                     alpha=2,
                     beta=0.75,
                     lambda_scale=2,
                     kappa=1,
                    nu=4.00)
        Z = np.array(nar_val).reshape(X.shape)
        ax.plot_wireframe(Y,X,Z,
                    lw=2.0)
        ax.plot_surface(Y,X,Z,alpha=0.3)
        ax.contourf(Y, X, Z, zdir='z', offset=0.05, cmap='Blues')
        Z_scatter = np.array(nar_val)
        ax.scatter(X_scatter,Y_scatter,Z_scatter,
                s=100,
                color='black',
                edgecolors='k',
                marker='h')
        ax.set_zlabel('$nAr^{*}(\Delta)$',fontsize=12)
        ax.set_xlabel('$\lambda_{ID}$',fontsize=12) 
        ax.set_ylabel('$\lambda_{ICGD}$',fontsize=12)
        ax.set_title('(d.vi) $\\nu=4.00$',fontsize=14)
        ax.view_init(elev=17., azim=30) 
        ax.set_zlim(0.05,0.65)

plt.show()
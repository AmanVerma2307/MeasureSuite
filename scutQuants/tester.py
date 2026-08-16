import itertools
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf
from sklearn.manifold import TSNE
from utils.parser import *
from model import getModel

args = parseArgs()

####### Loading Dataset
if(args.modelChoice != 'motionFormer'):
    X_dev = (np.load('./Datasets/SCUT/DGBQA-Seen/X_dev_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0'])
    X_dev_NonShuffled = (np.load('./Datasets/SCUT/DGBQA-Seen/X_dev_DGBQA_Seen_NonShuffled_SCUT.npz',allow_pickle=True)['arr_0'])
else:
    X_dev = (np.load('./Datasets/SCUT/DGBQA-Seen/X_dev_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0'])[:,:-1,:,:,:]
    X_dev_NonShuffled = (np.load('./Datasets/SCUT/DGBQA-Seen/X_dev_DGBQA_Seen_NonShuffled_SCUT.npz',allow_pickle=True)['arr_0'])[:,:-1,:,:,:]

y_dev = np.load('./Datasets/SCUT/DGBQA-Seen/y_dev_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']
y_dev_id = np.load('./Datasets/SCUT/DGBQA-Seen/y_dev_id_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']

###### Preparing One Hot Vectors
##### One Hot Encoding Creation
def get_ohot(vec):

    """
    INPUTS:-
    1) vec: Labels of shape (N,)

    OUPTUTS:-
    1) vec_ohot: Labels of shape (N,G); where G is the total classes
    """
    vec_ohot = np.zeros((vec.size,vec.max()+1))
    vec_ohot[np.arange(vec.size),vec] = 1
    return vec_ohot

##### Extracting One Hot Encoding
y_dev_id_ohot = get_ohot(y_dev_id)

##### Joint Label Creation
y_dev_final = np.append(np.append(np.reshape(y_dev,(y_dev.shape[0],1)),np.reshape(y_dev_id,(y_dev_id.shape[0],1)),axis=-1),
                            np.append(np.reshape(y_dev,(y_dev.shape[0],1)),y_dev_id_ohot,axis=-1),axis=-1)
print(y_dev.shape)

###### Testing the Model
strategy = tf.distribute.MirroredStrategy()

model = getModel(args,strategy)
model.summary()
model.load_weights('./Models/'+args.exp_name+'.h5')

####### Model Evaluation

##### Normalization Layer
def normalisation_layer(x):   
    return(tf.math.l2_normalize(x, axis=1, epsilon=1e-12))

###### Extracting Model Outputs
g_hgr, g_id, f_theta = model.predict(X_dev,batch_size=args.local_batch_size*strategy.num_replicas_in_sync)
f_theta_norm =  tf.keras.layers.Lambda(normalisation_layer)(f_theta)
f_theta_norm = f_theta_norm.numpy()
G_bar = np.matmul(f_theta_norm,f_theta_norm.T) # Gram-Matrix

g_hgr_nonshuffled, g_id_nonshuffled, f_theta_nonshuffled = model.predict(X_dev_NonShuffled,batch_size=args.local_batch_size*strategy.num_replicas_in_sync)
f_theta_nonshuffled_norm = tf.keras.layers.Lambda(normalisation_layer)(f_theta_nonshuffled)
f_theta_nonshuffled_norm = f_theta_nonshuffled_norm.numpy()
G_bar_nonshuffled = np.matmul(f_theta_nonshuffled_norm,f_theta_nonshuffled_norm.T) # Gram-Matrix

###### Accuracy Computations
##### Function to compute accuracy
def compute_accuracy(y_true,y_preds):
    
    """
    Function to compute accuracy in Sparse Categorical Prediction Style

    INPUTS:-
    1) y_true: Ground-trith sparse-categorical labels
    2) y_preds: Softmax layer outputs

    OUTPUTS:-
    1) acc_val: Accuracy Score
    """
    acc_val = tf.math.reduce_sum(tf.metrics.sparse_categorical_accuracy(y_true,y_preds))/y_true.shape[0]
    return acc_val

###### Accuracy Value
print('HGR Acc: '+str(compute_accuracy(y_dev,g_hgr))) # HGR Accuracy
print('ID Acc: '+str(compute_accuracy(y_dev_id,g_id))) # ID Accuracy

###### Saving Predictions
np.savez_compressed('./Predictions/'+args.exp_name+'.npz',g_hgr) # Saving Softmax predictions of shape: (N,G)

###### Softmax Heatmap
###### Computing Avg. Probability Scores
y_preds_hgr_probs = np.zeros((6,6))

##### Iterating over the Predicted Probabilites
for g_idx in range(6):

    g_prob = [] # List to store the Predicted Probabilties of the Current Class

    for idx, y_val_idx in enumerate(y_dev): # Iterating over the Dataset
        
        if(y_val_idx == g_idx):
            g_prob.append(g_hgr[idx])

    g_prob = np.around(np.mean(np.array(g_prob),axis=0),decimals=2)
    y_preds_hgr_probs[g_idx,:] = g_prob

print(y_preds_hgr_probs)

##### Plotting Heatmap

#### Heatmap Plotting Function
plt.rcParams["figure.figsize"] = [8,12]
def plot_heatmap(cm,filepath,classes,normalize=False,title='Avg. HGR Probabilities',cmap=plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_GramMatrix(cm,filepath,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

#### Heatmap Plotting
filepath='./Graphs/Softmax Heatmap/'+args.exp_name+'.png'
filepath_gram ='./Graphs/Gram Matrix/'+args.exp_name+'.png'
filepath_gram_ns ='./Graphs/Gram Matrix/'+args.exp_name+'_NonShuffled.png'
cm_plot_labels = ['Fist','Rotate to Fist','Catch and Release','Four Fingers','Bend Four Fingers','Fist Opening']
plot_heatmap(cm=np.around(y_preds_hgr_probs,2),filepath=filepath,classes=cm_plot_labels,normalize=False,title='Avg. Softmax Probability')
plot_GramMatrix(cm=G_bar,filepath=filepath_gram)
plot_GramMatrix(cm=G_bar_nonshuffled,filepath=filepath_gram_ns)

###### tSNE

##### Embedding Function
col_mean = np.nanmean(f_theta_norm, axis=0)
inds = np.where(np.isnan(f_theta_norm))
#print(inds)
f_theta_norm[inds] = np.take(col_mean, inds[1])

##### Saving Embeddings
np.savez_compressed('./Embeddings/'+args.exp_name+'.npz',f_theta_norm)

##### t-SNE Plots
#### t-SNE Embeddings
tsne_X_dev = TSNE(n_components=2,perplexity=30,learning_rate=10,n_iter=10000,n_iter_without_progress=50).fit_transform(f_theta_norm) # t-SNE Plots 

#### Plotting
plt.rcParams["figure.figsize"] = [12,8]
for idx,color_index in zip(list(np.arange(6)),["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]):
    plt.scatter(tsne_X_dev[y_dev == idx, 0],tsne_X_dev[y_dev == idx, 1],s=55,color=color_index,edgecolors='k',marker='h')
plt.legend(['Fist','Rotate to Fist','Catch and Release','Four Fingers','Bend Four Fingers','Fist Opening'],loc='best',prop={'size': 12})
#plt.grid(b='True',which='both')
plt.savefig('./Graphs/tSNE/'+args.exp_name+'.png')

####### Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf
import itertools
import argparse
import seaborn as sns
from sklearn.manifold import TSNE

####### Loading Dataset
###### Loading Arrays
X_train = np.load('./Datasets/SCUT/DGBQA-Seen/X_train_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']
X_dev = np.load('./Datasets/SCUT/DGBQA-Seen/X_dev_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']
y_train = np.load('./Datasets/SCUT/DGBQA-Seen/y_train_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']
y_dev = np.load('./Datasets/SCUT/DGBQA-Seen/y_dev_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']
y_train_id = np.load('./Datasets/SCUT/DGBQA-Seen/y_train_id_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']
y_dev_id = np.load('./Datasets/SCUT/DGBQA-Seen/y_dev_id_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']

X_dev_nonshuffled = np.load('./Datasets/SCUT/DGBQA-Seen/X_dev_DGBQA_Seen_NonShuffled_SCUT.npz',allow_pickle=True)['arr_0'] # Non-Shuffled

####### Model Arguments and Hyperparameters
parser = argparse.ArgumentParser()

parser.add_argument("--lambda_id",
                    type=float,
                    help="Scaling Value of ID Loss")
parser.add_argument("--lambda_cgid",
                    type=float,
                    help="Scaling Value of CGID Loss")
parser.add_argument("--local_batch_size",
                    type=int,
                    help="Batch Size to used for a device")
parser.add_argument("--exp_name",
                    type=str,
                    help="Name of the Experiment being run, will be used saving the model and correponding outputs")

args = parser.parse_args()

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
y_train_id_ohot = get_ohot(y_train_id)
y_dev_id_ohot = get_ohot(y_dev_id)

##### Joint Label Creation
y_train_final = np.append(np.append(np.reshape(y_train,(y_train.shape[0],1)),np.reshape(y_train_id,(y_train_id.shape[0],1)),axis=-1),
                            np.append(np.reshape(y_train,(y_train.shape[0],1)),y_train_id_ohot,axis=-1),axis=-1)
y_dev_final = np.append(np.append(np.reshape(y_dev,(y_dev.shape[0],1)),np.reshape(y_dev_id,(y_dev_id.shape[0],1)),axis=-1),
                            np.append(np.reshape(y_dev,(y_dev.shape[0],1)),y_dev_id_ohot,axis=-1),axis=-1)
print(y_train_final.shape,y_dev.shape)

####### Distribution Strategy
strategy = tf.distribute.MirroredStrategy()

####### Model Making

###### Video Vision Transformer 

##### Tubelet Embedding
class Tubelet_Embedding(tf.keras.layers.Layer):

    def __init__(self, embed_dim, patch_size):

        #### Defining Essentials
        super().__init__()
        self.embed_dim = embed_dim # Embedding Dimensions 
        self.patch_size = patch_size # A tuple of dimensions - (p_t,p_h,p_w), with each corresponding to patch dimensions

        #### Defining Layers
        self.embedding_layer =  tf.keras.layers.Conv3D(filters=self.embed_dim,
                                                        kernel_size=self.patch_size,
                                                        strides=self.patch_size,
                                                        padding="VALID") # Tubelet Patch and Embedding Creation Layer
        self.flatten =  tf.keras.layers.Reshape((-1,self.embed_dim)) # Layer to Flatten the Patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size
        })

    def call(self,X_in):

        """
        Layer to Project the input spatio-temporal sequence into Tubelet Tokens

        INPUTS:-
        1) X_in: Input video sequence of dimensions (T,H,W,C)

        OUTPUTS:-
        1) X_o: Tubelet Embeddings of shape (n_t*n_h*n_w,embed_dim)
        
        """
        #### Tubelet Embedding Creation
        X_o = self.embedding_layer(X_in) # Embedding Layer
        X_o = self.flatten(X_o) # Flattening Input

        return X_o

###### Positional Embedding
class PositionEmbedding(tf.keras.layers.Layer):
    
    def __init__(self, maxlen, embed_dim):

        #### Defining Essentials
        super().__init__()
        self.maxlen = maxlen # Maximum Signal Length
        self.embed_dim = embed_dim # Input Embedding Dimensions

        #### Defining Layers
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen, 
            'embed_dim': self.embed_dim 
        })
        return config 

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

###### Encoder Block
class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads, dff_dim, rate=0.1):

        #### Defining Essentials
        super().__init__()
        self.d_model = d_model # Embedding Dimensions of the Encoder Layer
        self.num_heads = num_heads # Number of Self-Attention Heads
        self.dff_dim = dff_dim # Projection Dimensions of Feed-Forward Network
        self.rate = rate # Dropout Rate

        #### Defining Layers
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,key_dim=self.d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff_dim, activation="relu"),
            tf.keras.layers.Dense(self.d_model),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model, 
            'num_heads': self.num_heads, 
            'dff_dim': self.dff_dim,
            'rate': self.rate
        })
        return config 

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)  # self-attention layer
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # layer norm
        ffn_output = self.ffn(out1)  #feed-forward layer
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # layer norm
    
####### Cross-Gesture Identity-Disentanglement Loss

###### Mask Generation

##### Positive Mask
@tf.function
def get_positive_mask(labels):
    """
    Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(labels.shape[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks``
    mask = tf.logical_and(indices_not_equal, labels_equal)

    # label-mask
    one_vec = tf.ones_like(tf.reshape(labels,(labels.shape[0],1)))
    zero_mask = tf.linalg.matmul(one_vec,tf.reshape(labels,(labels.shape[0],1)),transpose_b=True)

    # Mask Generation
    mask = tf.logical_and(mask, tf.cast(zero_mask,dtype=tf.bool))

    return mask
    
##### Negative Mask - Different Mask
@tf.function
def get_negative_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask

###### Loss Function
class CG_ID_Loss(tf.keras.losses.Loss):

    """
    Loss to Enforce Identity level gesture disentanglement.

    INPUTS:
    1) N: Batch-Size
    2) d: Embedding Dimensions
    3) I: Total Identities
    4) G: Total Gestures
    """

    def __init__(self,N,d,I,G):
        
        ##### Defining Essentials
        super().__init__()
        self.N = N # Batch Size
        self.d = d # Embedding Dimensions
        self.I = I # Total Identities
        self.G = G # Total Gestures

    def get_config(self):

        config = super().get_config.copy()
        config.update({
            'N':self.N,
            'd':self.d,
            'I':self.I,
            'G':self.G
        })
        return config
    
    @tf.function
    def call(self,y_stash,f_theta):

        """
        Enforcing Gramian Matrix to become Identity Matrix, considering L2-Normalized embeddings. 

        INPUTS:-  
        1) f_theta: Final Embeddings of the embedder; shape=(self.N,self.d)
        2) y_stash: Vector List:[y_hgr,y_id] with y_hgr.shape=(N,) and y_id being one-hot encoded of shape (self.N,self.I)

        OUTPUTS:-
        1) loss_batch: Total L-CGID for the Batch
        """
        ##### Separating Labels
        y_hgr = y_stash[:,0] # HGR Labels - Useful for Boolean Mask Creation
        y_id = y_stash[:,1:] # Identity Labels - Useful for Disentangling Terms Estimation        

        ##### L2-Normalization
        f_theta = tf.math.l2_normalize(f_theta,axis=1)

        ##### Gramian Matrix Formation
        G_bar = tf.linalg.matmul(f_theta,f_theta,transpose_b=True)

        ##### Gramian-Matrix Positive Mask
        zero_matrix = tf.zeros_like(G_bar) # Matrix of all zeros to compare with Gramian Matrix
        Gamma_bar = tf.cast(tf.math.greater_equal(G_bar,zero_matrix),dtype=tf.float32) # Mask for all the negative values

        ##### Different Gesture Mask Computation
        delta_bar = get_negative_mask(y_hgr)

        ##### Lower Triangular Matrix
        LT_Mask = tf.linalg.band_part(tf.ones(shape=G_bar.shape),0,-1) # Lower Triangular Matrix

        ##### Loss Computation
        #### Defining Essentials
        Loss_CG_ID = 0 # Loss for the Current Batch

        #### Iterating over the Identities
        for sub_idx in range(self.I):

            y_id_curr = y_id[:,sub_idx] # Extracting labels for the current identity
            delta_curr = get_positive_mask(y_id_curr) # Extracting positive mask of the current identity
            Loss_CG_ID_curr = tf.math.reduce_sum(tf.math.multiply(Gamma_bar,tf.math.abs(tf.math.multiply(tf.math.multiply(tf.cast(LT_Mask,dtype=tf.float32),tf.cast(delta_bar,dtype=tf.float32)),
                                                                                      tf.math.multiply(tf.cast(delta_curr,dtype=tf.float32),G_bar)))))
            Normalization_Factor = tf.math.reduce_sum(tf.math.multiply(Gamma_bar,tf.math.multiply(tf.math.multiply(tf.cast(LT_Mask,dtype=tf.float32),tf.cast(delta_bar,dtype=tf.float32)),
                                                                       tf.cast(delta_curr,dtype=tf.float32)))) 
            Loss_CG_ID = Loss_CG_ID + (Loss_CG_ID_curr/(Normalization_Factor+1))

        return Loss_CG_ID/self.I       

###### Custom Model Checkpointing
class ModelCheckpointing_Loss(tf.keras.callbacks.Callback):

    """
     Callback to save the model with least validation loss
    """

    def __init__(self,filepath):
        
        ##### Defining Essentials    
        super(ModelCheckpointing_Loss, self).__init__()
        self.best_loss = np.inf # Initializing with Infinite Loss
        self.filepath = filepath # Path of the File wherein weights are to be saved

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        #### Logging Current Values
        loss_curr = logs['val_loss']

        #### Saving Weights
        if(loss_curr < self.best_loss):
            self.model.save_weights(self.filepath) # Saving Model
            self.best_loss = loss_curr # Updating current loss

        else:
            return

####### Model Training
###### Defining Layers and Model

###### Defining Essentials
T = 63
H = 64
W = 64
C_rdi = 1
num_layers = 2
d_model = 32
num_heads = 16
dff_dim = 128
p_t = 5
p_h = 5
p_w = 5
n_t = (((T - p_t)//p_t)+1)
n_h = (((H - p_h)//p_h)+1)
n_w = (((W - p_w)//p_w)+1)
max_seq_len = int(n_t*n_h/2*n_w/2)
pe_input = int(n_t*n_h/2*n_w/2)
rate = 0.3

###### Defining Layers

##### Convolutional Layers

#### Res3DNet
conv11_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
conv12_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
conv13_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
maxpool_1 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2))

conv21_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
conv22_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
conv23_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')

##### ViViT
tubelet_embedding_layer = Tubelet_Embedding(d_model,(p_t,p_h,p_w))
positional_embedding_encoder = PositionEmbedding(max_seq_len,d_model)
enc_block_1 = Encoder(d_model,num_heads,dff_dim,rate)
enc_block_2 = Encoder(d_model,num_heads,dff_dim,rate)

###### Defining Model
with strategy.scope():

    ##### Input Layer
    Input_Layer = tf.keras.layers.Input(shape=(T,H,W,C_rdi))

    ##### Conv Layers

    #### Res3DNet
    ### Residual Block - 1
    conv11_rdi = conv11_rdi(Input_Layer) 
    conv12_rdi = conv12_rdi(conv11_rdi)
    conv13_rdi = conv13_rdi(conv12_rdi)
    conv13_rdi = tf.keras.layers.Add()([conv13_rdi,conv11_rdi])
    conv13_rdi = maxpool_1(conv13_rdi)

    ### Residual Block - 2
    conv21_rdi = conv21_rdi(conv13_rdi)
    conv22_rdi = conv22_rdi(conv21_rdi)
    conv23_rdi = conv23_rdi(conv22_rdi)
    conv23_rdi = tf.keras.layers.Add()([conv23_rdi,conv21_rdi])

    #####  ViViT
    tubelet_embedding = tubelet_embedding_layer(conv23_rdi)
    tokens = positional_embedding_encoder(tubelet_embedding)
    enc_block_1_op = enc_block_1(tokens)
    enc_block_2_op = enc_block_2(enc_block_1_op)

    ##### Output Layer
    gap_op = tf.keras.layers.GlobalAveragePooling1D()(enc_block_2_op)
    dense1 = tf.keras.layers.Dense(32,activation='relu')(gap_op)

    #### HGR Output
    dense2_hgr = tf.keras.layers.Dense(6,activation='softmax')(dense1)

    #### ID Output
    dense2_id = tf.keras.layers.Dense(143,activation='softmax')(dense1)

    ###### Model Definition
    model = tf.keras.models.Model(inputs=Input_Layer,outputs=[dense2_hgr,dense2_id,dense1])
    model.load_weights('./Models/'+args.exp_name+'.h5')
#model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss=['sparse_categorical_crossentropy','sparse_categorical_crossentropy',l_CGID],loss_weights=[1,1,1],metrics='accuracy')
model.summary()
#tf.keras.utils.plot_model(model)

####### Model Evaluation

##### Normalization Layer
def normalisation_layer(x):   
    return(tf.math.l2_normalize(x, axis=1, epsilon=1e-12))

###### Extracting Model Outputs
g_hgr, g_id, f_theta = model.predict(X_dev,batch_size=8*strategy.num_replicas_in_sync)
f_theta_norm =  tf.keras.layers.Lambda(normalisation_layer)(f_theta)
f_theta_norm = f_theta_norm.numpy()
G_bar = np.matmul(f_theta_norm,f_theta_norm.T) # Gram-Matrix

g_hgr_nonshuffled, g_id_nonshuffled, f_theta_nonshuffled = model.predict(X_dev_nonshuffled,batch_size=8*strategy.num_replicas_in_sync)
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

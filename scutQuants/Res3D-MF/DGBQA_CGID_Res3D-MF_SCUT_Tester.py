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
X_train = (np.load('./Datasets/SCUT/DGBQA-Seen/X_train_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0'])[:,:-1,:,:,:]
X_dev = (np.load('./Datasets/SCUT/DGBQA-Seen/X_dev_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0'])[:,:-1,:,:,:]
y_train = np.load('./Datasets/SCUT/DGBQA-Seen/y_train_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']
y_dev = np.load('./Datasets/SCUT/DGBQA-Seen/y_dev_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']
y_train_id = np.load('./Datasets/SCUT/DGBQA-Seen/y_train_id_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']
y_dev_id = np.load('./Datasets/SCUT/DGBQA-Seen/y_dev_id_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']

X_dev_NonShuffled = (np.load('./Datasets/SCUT/DGBQA-Seen/X_dev_DGBQA_Seen_NonShuffled_SCUT.npz',allow_pickle=True)['arr_0'])[:,:-1,:,:,:] # Non-Shuffled

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

###### MotionFormer

####### Patch Embedding Layer
class Patch_Embedding(tf.keras.layers.Layer):

    def __init__(self, T, embed_dim, patch_size):

        #### Defining Essentials
        super().__init__()
        self.T = T # Number of Frames
        self.embed_dim = embed_dim # Embedding Dimensions 
        self.patch_size = patch_size # A tuple of dimensions - (p_t,p_h,p_w), with each corresponding to patch dimensions

        #### Defining Layers
        self.embedding_layer = tf.keras.layers.Conv2D(filters=self.embed_dim,
                                                        kernel_size=self.patch_size,
                                                        strides=self.patch_size,
                                                        padding="VALID") # Tubelet Patch and Embedding Creation Layer
        self.flatten = tf.keras.layers.Reshape((-1,self.embed_dim)) # Layer to Flatten the Patches to Dimension (ST,D)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'T': self.T,
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

####### Positional Embedding Layer
class PositionEmbedding(tf.keras.layers.Layer):
    
    def __init__(self, maxlen_spatial, num_frames, embed_dim):

        #### Defining Essentials
        super().__init__()
        self.maxlen_spatial = maxlen_spatial # Maximum Spatial Length
        self.num_frames = num_frames # Number of Frames
        self.embed_dim = embed_dim # Input Embedding Dimensions

        #### Defining Layers
        self.pos_emb = tf.keras.layers.Embedding(input_dim=self.maxlen_spatial*self.num_frames, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen_spatial': self.maxlen_spatial,
            'num_frames': self.num_frames, 
            'embed_dim': self.embed_dim 
        })
        return config 

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen_spatial*self.num_frames, delta=1) # Position Range
        positions = self.pos_emb(positions) # Embedding the Positional Embedding
        #positions = tf.keras.layers.Reshape((self.num_frames,self.maxlen_spatial,self.embed_dim))(positions) # Reshaping the Dimensions
        return x + positions # Addition of Positional EmbeddingsS

###### Multi-Head Inter-Frame Spatial Attention
class MIFSA(tf.keras.layers.Layer):

    """
    Multi-Head Inter-Frame Spatial Attention Module
    """

    def __init__(self,num_heads,d_model,T,S):

        ##### Defining Essentials
        super().__init__()
        self.num_heads = num_heads # Number of Attention Heads
        self.d_model = d_model # Model Embedding Dimensions: Soft Attention requires d_model // num_heads = 0
        self.T = T # Number of Frames
        self.S = S # Maximum Length of spatial tokens
        self.depth = self.d_model // self.num_heads # Embedding Dimensions per Head

        ##### Defining Layers
        self.query_dense = tf.keras.layers.Dense(self.d_model) # Query Embedding Layer
        self.key_dense = tf.keras.layers.Dense(self.d_model) # Key Embedding Layer
        self.value_dense = tf.keras.layers.Dense(self.d_model) # Value Embedding Layer
        self.concat_dense = tf.keras.layers.Dense(self.d_model) # Multi-Head Dense Layer

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'T': self.T,
            'S': self.S,
            'depth': self.depth
        })
        return config 

    def split_heads(self, inputs):

        """
        Function to split the head

        INPUTS:-
        1) inputs: Tokens of shape (N,TS,D)

        OUTPUTS:-
        1) inputs: Tokens reshaped to (N,num_heads,TS,depth)
        """

        inputs = tf.keras.layers.Reshape((-1,self.num_heads,self.depth))(inputs)
        inputs = tf.transpose(inputs,perm=[0,2,1,3])
        return inputs

    def scaled_dot_product_attention(self, q, k, v):

        """
        Function to Compute Dot-Product Attention Modulation

        INPUTS:-
        1) q: Query of shape (N,num_heads,TS,depth)
        2) k: Key of shape (N,num_heads,S,depth)
        3) v: Value of shape (N,num_heads,S,depth)

        OUTPUTS:-
        1) output: Dot-Product Attention Output of shape (N)
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True) # Attention Matrix
        dk = tf.cast(tf.shape(k)[-1], tf.float32) # Scaling Factor
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) # Attention Scaling
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # Attention Weights: Softmax Activation along 'S' axis - shape -> (N,num_heads,TS,S)
        output = tf.matmul(attention_weights, v) # Attention Multiplication: shape -> (N,num_heads,TS,depth)
        return output

    def call(self,X):

        """
        Multi-Head Inter-Frame Spatial Attention Module

        INPUTS:-
        1) X: Tokens of shape (N,TS,D)

        OUTPUTS:-
        1) X_mifsa: Attention Output of shape (N,TS,T,D)
        """

        ##### Defining Essentials
        attn_op_list = [] # List to store per frame attention output

        ##### Query Generation
        Q_misfa = self.query_dense(X) # shape -> (N,TS,D)
        Q_misfa = self.split_heads(Q_misfa) # shape -> (N,num_heads,TS,depth)

        ##### Reshaping the Input
        X_rshp = tf.keras.layers.Reshape((self.T,self.S,self.d_model))(X) # shape -> (N,T,S,d_model)

        ##### Iterating over the Temporal Frames for Inter-Frame Spatial Attention
        for t_prime in range(self.T):

            #### Selecting Spatial tokens of frame with index t_prime
            X_t_prime = tf.keras.layers.Reshape((self.S,self.d_model))(X_rshp[:,t_prime,:,:]) # shape -> (N,S,d_model)

            #### Key and Value Generation
            ### Key
            K_misfa = self.key_dense(X_t_prime) # shape -> (N,S,d_model)
            K_misfa = self.split_heads(K_misfa) # shape -> (N,num_heads,S,depth)

            ### Value
            V_misfa = self.value_dense(X_t_prime) # shape -> (N,S,d_model)
            V_misfa = self.split_heads(V_misfa) # shape -> (N,num_heads,S,depth)

            ### Attention Output
            attn_op_t_prime = self.scaled_dot_product_attention(Q_misfa,K_misfa,V_misfa) # shape -> (N,num_heads,TS,depth)
            attn_op_t_prime = tf.transpose(attn_op_t_prime,perm=[0,2,1,3]) # shape -> (N,TS,num_heads,depth)
            attn_op_t_prime = tf.keras.layers.Reshape((-1,self.d_model))(attn_op_t_prime) # shape -> (N,TS,d_model)
            attn_op_t_prime = self.concat_dense(attn_op_t_prime) # shape -> (N,TS,d_model)

            attn_op_list.append(attn_op_t_prime) # Attention Output 

        ##### Stacking and Reshaping per-frame Attention Outputs
        X_mifsa = tf.stack(attn_op_list,axis=-1) # Stacking Operation: shape -> (N,TS,d_model,T)
        X_mifsa = tf.transpose(X_mifsa,perm=[0,1,3,2]) # Reshaping Operation: shape -> (N,TS,T,d_model)

        return X_mifsa
    
###### Temporal Trajectory Aggregation Attention
class TTAA(tf.keras.layers.Layer):

    """
    Temporal Trajectory Aggregation Attention
    """

    def __init__(self,num_heads,d_model,T,S):

        ##### Defining Essentials
        super().__init__()
        self.num_heads = num_heads # Number of Attention Heads
        self.d_model = d_model # Model Embedding Dimensions: Soft Attention requires d_model // num_heads = 0
        self.T = T # Number of Frames
        self.S = S # Maximum Length of spatial tokens
        self.depth = self.d_model // self.num_heads # Embedding Dimensions per Head

        ##### Defining Layers
        self.query_dense = tf.keras.layers.Dense(self.d_model) # Query Embedding Layer
        self.key_dense = tf.keras.layers.Dense(self.d_model) # Key Embedding Layer
        self.value_dense = tf.keras.layers.Dense(self.d_model) # Value Embedding Layer
        self.concat_dense = tf.keras.layers.Dense(self.d_model) # Multi-Head Dense Layer

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'T': self.T,
            'S': self.S,
            'depth': self.depth
        })
        return config 

    def split_heads_q(self, inputs):

        """
        Function to split the heads for Query Tokens

        INPUTS:-
        1) inputs: Tokens of shape (N,S,1,d_model)

        OUTPUTS:-
        1) inputs: Tokens reshaped to (N,num_heads,S,1,depth)
        """

        inputs = tf.keras.layers.Reshape((self.S,1,self.num_heads,self.depth))(inputs)
        inputs = tf.transpose(inputs,perm=[0,3,1,2,4])
        return inputs
    
    def split_heads_kv(self, inputs):

        """
        Function to split the heads for Key/Value Tokens

        INPUTS:-
        1) inputs: Tokens of shape (N,S,T,d_model)

        OUTPUTS:-
        1) inputs: Tokens reshaped to (N,num_heads,S,T,depth)
        """

        inputs = tf.keras.layers.Reshape((self.S,self.T,self.num_heads,self.depth))(inputs)
        inputs = tf.transpose(inputs,perm=[0,3,1,2,4])
        return inputs

    def scaled_dot_product_attention(self, q, k, v):

        """
        Function to Compute Dot-Product Attention Modulation

        INPUTS:-
        1) q: Query of shape (N,num_heads,S,1,depth)
        2) k: Key of shape (N,num_heads,S,T,depth)
        3) v: Value of shape (N,num_heads,S,T,depth)

        OUTPUTS:-
        1) output: Dot-Product Attention Output of shape (N,num_heads,S,1,depth)
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True) # Attention Matrix: shape -> (N,num_heads,S,1,T)
        dk = tf.cast(tf.shape(k)[-1], tf.float32) # Scaling Factor
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) # Attention Scaling
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # Attention Weights: Attention along Temporal Axis
        output = tf.matmul(attention_weights, v) # Attention Multiplication: shape -> (N,num_heads,S,1,depth)
        return output

    def call(self,X):

        """
        Temporal Trajectory Aggregation Attention

        INPUTS:-
        1) X: Tokens of shape (N,TS,T,D)

        OUTPUTS:-
        1) X_ttaa: Attention Output of shape (N,TS,D)
        """

        ##### Defining Essentials
        attn_op = [] # List to store Attention output per temporal index

        ##### Input Reshaping Operation
        X_rshp = tf.keras.layers.Reshape((self.T,self.S,self.T,-1))(X) # shape -> (N,T,S,T,d_model)

        ##### Iteration over time-index
        for t_prime in range(self.T): 

            #### Temporal Tokens Collectin
            X_t_prime = X_rshp[:,t_prime,:,:,:] # shape -> (N,S,T,d_model)

            #### Query Generation
            ### Query Token Selection
            X_t_prime_q = X_t_prime[:,:,t_prime,:] # shape -> (N,S,d_model)
            X_t_prime_q = tf.keras.layers.Reshape((self.S,1,self.d_model))(X_t_prime_q) # shape -> (N,S,1,d_model)

            ### Query Generation
            Q_t_prime = self.query_dense(X_t_prime_q) # shape -> (N,S,1,d_model)
            Q_t_prime = self.split_heads_q(Q_t_prime) # shape -> (N,num_heads,S,1,depth)

            #### Key and Value Generation
            ### Key
            K_t_prime = self.key_dense(X_t_prime) # shape -> (N,S,T,d_model)
            K_t_prime = self.split_heads_kv(K_t_prime) # shape -> (N,num_heads,S,T,depth)

            ### Value
            V_t_prime = self.value_dense(X_t_prime) # shape -> (N,S,T,d_model)
            V_t_prime = self.split_heads_kv(V_t_prime) # shape -> (N,num_heads,S,T,depth)

            #### Attention Output Generation
            O_t_prime = self.scaled_dot_product_attention(Q_t_prime,K_t_prime,V_t_prime) # shape -> (N,num_heads,S,1,depth)
            O_t_prime = tf.transpose(O_t_prime,perm=[0,2,3,1,4]) # shape -> (N,S,1,num_heads,depth)
            O_t_prime = tf.keras.layers.Reshape((-1,self.d_model))(O_t_prime) # shape -> (N,S,d_model)
            O_t_prime = self.concat_dense(O_t_prime) # shape -> (N,S,d_model)

            attn_op.append(O_t_prime) # Accumulating outputs for temporal indices

        ##### Stacking and Reshaping per-Temporal Index Attention Outputs
        X_ttaa = tf.stack(attn_op,axis=-1) # Stacking Operation: shape -> (N,S,d_model,T)
        X_ttaa = tf.transpose(X_ttaa,perm=[0,3,1,2]) # Arrangement Operation: shape -> (N,T,S,d_model)
        X_ttaa = tf.keras.layers.Reshape((-1,self.d_model))(X_ttaa) # Reshape Operation: shape -> (N,TS,d_model)

        return X_ttaa
    
###### MotionFormer Encoder
class MotionFormer_Encoder(tf.keras.layers.Layer):
    
    def __init__(self, num_heads, d_model, dff_dim, T, S, rate=0.1):

        #### Defining Essentials
        super().__init__()
        self.num_heads = num_heads # Number of Self-Attention Heads
        self.d_model = d_model # Embedding Dimensions of the Encoder Layer
        self.dff_dim = dff_dim # Projection Dimensions of Feed-Forward Network
        self.T = T # Number of Temporal Frames
        self.S = S # Maximum Spatial Length
        self.rate = rate # Dropout Rate

        #### Defining Layers
        self.mifsa = MIFSA(self.num_heads,self.d_model,self.T,self.S) # MIFSA Module
        self.ttaa = TTAA(self.num_heads,self.d_model,self.T,self.S) # TTAA Module
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
            'num_heads': self.num_heads,
            'd_model': self.d_model,  
            'dff_dim': self.dff_dim,
            'T': self.T,
            'S': self.S,
            'rate': self.rate
        })
        return config 

    def call(self, inputs, training):

        """
        MotionFormer Encoder Block: Transformer Mechanism with Trajectory Attention

        INPUTS:-
        1) inputs: Input Tokens of shape (N,TS,d_model)

        OUTPUTS:-
        1) output: Output Tokens of shape (N,TS,d_model)
        """
        attn_output = self.mifsa(inputs) # MIFSA Layer: shape -> (N,TS,T,d_model)
        attn_output = self.ttaa(attn_output) # TTAA Layer: shape -> (N,TS,d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # layer norm
        ffn_output = self.ffn(out1)  #feed-forward layer
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)  # layer norm: shape -> (N,TS,d_model)
        return output
    
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
T = 62
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
n_t = int(T/2)
n_h = (((H - p_h)//p_h)+1)
n_w = (((W - p_w)//p_w)+1)
S = int(n_h/2*n_w/2)
max_seq_len = n_t*(n_h/2*n_w/2)
pe_input = n_t*(n_h/2*n_w/2)
rate = 0.3

###### Defining Layers

##### Convolutional Layers

#### Res3DNet
conv11_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
conv12_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
conv13_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
maxpool_1 = tf.keras.layers.MaxPool3D(pool_size=(2,2,2))

conv21_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
conv22_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
conv23_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')

##### ViViT
patch_embedding_layer = Patch_Embedding(n_t,d_model,(p_h,p_w))
positional_embedding_encoder = PositionEmbedding(S,n_t,d_model)
enc_block_1 = MotionFormer_Encoder(num_heads,d_model,dff_dim,n_t,S,rate)
enc_block_2 = MotionFormer_Encoder(num_heads,d_model,dff_dim,n_t,S,rate)

###### Defining Model

with strategy.scope(): # Model Declaration under the scope of Mirrored Strategy

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
    tubelet_embedding = patch_embedding_layer(conv23_rdi)
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

g_hgr_nonshuffled, g_id_nonshuffled, f_theta_nonshuffled = model.predict(X_dev_NonShuffled,batch_size=8*strategy.num_replicas_in_sync)
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

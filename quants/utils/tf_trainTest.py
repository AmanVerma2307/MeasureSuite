####### Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf
import os                                                                                                         
import gc
import math
import pydot
import wandb
from sklearn.utils import shuffle
 
####### Loading Dataset
X_train = (np.load('./data/soli/X_train_Seen-IAR-NonShuffled_SOLI.npz')['arr_0'])
X_dev = (np.load('./data/soli/X_dev_Seen-IAR-NonShuffled_SOLI.npz')['arr_0'])
y_train = np.load('./data/soli/y_train_Seen-IAR-NonShuffled_SOLI.npz')['arr_0']
y_dev = np.load('./data/soli/y_dev_Seen-IAR-NonShuffled_SOLI.npz')['arr_0']
y_train_id = np.load('./data/soli/y_train_id_Seen-IAR-NonShuffled_SOLI.npz')['arr_0']
y_dev_id = np.load('./data/soli/y_dev_id_Seen-IAR-NonShuffled_SOLI.npz')['arr_0']

X_train, y_train, y_train_id = shuffle(X_train,y_train,y_train_id,random_state=12)
X_dev, y_dev, y_dev_id = shuffle(X_dev, y_dev, y_dev_id,random_state=12)

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
T = 40
H = 32
W = 32
C_rdi = 4
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
max_seq_len = n_t*n_h*n_w
pe_input = n_t*n_h*n_w
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

##### Input Layer
Input_Layer = tf.keras.layers.Input(shape=(T,H,W,C_rdi))

##### Conv Layers

#### Res3DNet
### Residual Block - 1
conv11_rdi = conv11_rdi(Input_Layer)
conv12_rdi = conv12_rdi(conv11_rdi)
conv13_rdi = conv13_rdi(conv12_rdi)
conv13_rdi = tf.keras.layers.Add()([conv13_rdi,conv11_rdi])
#conv13_rdi = maxpool_1(conv13_rdi)

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
dense2_hgr = tf.keras.layers.Dense(11,activation='softmax')(dense1)

#### ID Output
dense2_id = tf.keras.layers.Dense(10,activation='softmax')(dense1)

###### Compiling Model
model = tf.keras.models.Model(inputs=Input_Layer,outputs=[dense2_hgr,dense2_id])
model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss=['sparse_categorical_crossentropy','sparse_categorical_crossentropy'],loss_weights=[0,1.0],metrics='accuracy')
model.summary()
#tf.keras.utils.plot_model(model)

##### Defining Callbacks 
filepath= "./models/ID_Res3D-ViViT_1_SOLI.h5" 
checkpoint = ModelCheckpointing_Loss(filepath) 

###### Training the Model
history = model.fit(X_train,(y_train,y_train_id),epochs=150,batch_size=32,
                validation_data=(X_dev,(y_dev,y_dev_id)), validation_batch_size=32,
                   callbacks=checkpoint)

##### Saving Training Metrics
#np.save('./Model History/DGBQA_HGR-ID_Res3D-ViViT_1pt5_SOLI.npy',history.history)

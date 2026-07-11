import time
import argparse
import numpy as np
import tensorflow as tf
from models.motionFormer import *
from loss.icgd import *

####### Loading Dataset
X_train = (np.load('./Datasets/SCUT/DGBQA-Seen/X_train_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0'])[:,:-1,:,:,:]
X_dev = (np.load('./Datasets/SCUT/DGBQA-Seen/X_dev_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0'])[:,:-1,:,:,:]
y_train = np.load('./Datasets/SCUT/DGBQA-Seen/y_train_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']
y_dev = np.load('./Datasets/SCUT/DGBQA-Seen/y_dev_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']
y_train_id = np.load('./Datasets/SCUT/DGBQA-Seen/y_train_id_DGBQA_Seen_SCUT.npz',allow_pickle=True)['arr_0']
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
parser.add_argument("--numEpochs",
                    type=int,
                    default=100,
                    help="Number of epochs to run")

args = parser.parse_args()

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
positional_embedding_encoder = positionalEmbedding_mf(S,n_t,d_model)
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
    #model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss=['sparse_categorical_crossentropy','sparse_categorical_crossentropy',l_CGID],loss_weights=[1,1,1],metrics='accuracy')
model.summary()
#tf.keras.utils.plot_model(model)

##### Defining Callbacks
#filepath= "./Models/DGBQA_Vanilla+CGID_Res3D-ViViT_1-1-1_IAR_SOLI.h5"
#checkpoint = ModelCheckpointing_Loss(filepath) 

###### Training the Model

##### Defining Essentials
#### Training Heuristics
num_epochs = args.numEpochs
batch_size = args.local_batch_size # Local Batch Size
G_Total = 6 # Total Gestures
I_Total = 143 # Total Identites
lambda_id = args.lambda_id # Scaling weights for ID-Loss
lambda_cgid = args.lambda_cgid # Scaling weights for CG-ID Loss
train_loss = [] # List to store training loss
train_acc_hgr = [] # List to store training HGR accuracy
train_acc_id = [] # List to store training ID accuracy
val_loss = [] # List to store validation loss  
val_acc_hgr = [] # List to store validation HGR accuracy
val_acc_id = [] # List to store validation ID accuracy
filepath = "./Models/"+args.exp_name+'.h5'
Total_Training_Examples = X_train.shape[0]
Total_Val_Examples = X_dev.shape[0]
val_loss_best = 1e6 # Arbitrary value to monitor loss
val_acc_hgr_best = 0.0 # Arbitraty valu eto montor accuracy
val_loss_margin = 0.5 # Margin for validation loss
train_loss_collate = [] # List to store all the training losses collectively
val_loss_collate = [] # List to store all the validation losses collectively

#### Loss Functions
with strategy.scope():
    loss_func_hgr = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
    loss_func_id  = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
    loss_func_cgid = CG_ID_Loss(batch_size,d_model,I_Total,G_Total)
    loss_func_cgid.reduction = tf.keras.losses.Reduction.SUM

#### Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

##### Dataset Definition
train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train_final))
#train_dataset = train_dataset.shuffle(buffer_size=tf.int64(Total_Training_Examples))
train_dataset = train_dataset.batch(batch_size*strategy.num_replicas_in_sync) # Using Distributed Batch-Size
train_dataset = strategy.experimental_distribute_dataset(train_dataset) # Distributed Training-Data Iterator

val_dataset = tf.data.Dataset.from_tensor_slices((X_dev,y_dev_final))
#val_dataset = val_dataset.shuffle(buffer_size=tf.int64(Total_Val_Examples))
val_dataset = val_dataset.batch(batch_size*strategy.num_replicas_in_sync) # Using Distributed Batch-Sizer
val_dataset = strategy.experimental_distribute_dataset(val_dataset) # Distributed Val-Data Iterator

##### Training Function
#### Device Train-Step
@tf.function
def train_step(X_batch,y_batch,lambda_id,lambda_cgid,optimizer):

    """
    Function to Train the Model for a batch

    INPUTS:-
    1) X_batch: Batch of Tensor Arrays comprising Inputs
    2) y_batch: Batch of Tensor Arrays comprising Labels: [y_hgr,y_id,y_cgid]
    3) lambda_id: Scaling factor for ID-Loss
    4) lambda_cgid: Scaling factor for CG-ID Loss
    5) optimizer: Tensorflow optimizer object

    OUTPUTS:-
    1) loss_batch: Loss value for the batch*Number_of_samples_in_batch
    2) acc_batch_hgr: (HGR Accuracy value for the batch)*Number_of_samples_in_batch
    3) acc_batch_id: (ID Accuracy value for the batch)*Number_of_samples_in_batch
    """

    #### Unpacking labels
    y_hgr_batch = y_batch[:,0]
    y_id_batch = y_batch[:,1]
    y_cgid_batch = y_batch[:,2:]

    #### Gradient Computation
    with tf.GradientTape() as tape:

        ### Output Computation
        g_hgr_batch, g_id_batch, f_theta_batch = model(X_batch)

        ### Loss Computations
        loss_hgr_batch = loss_func_hgr(y_hgr_batch,g_hgr_batch)
        loss_id_batch = loss_func_id(y_id_batch,g_id_batch)
        loss_cgid_batch = loss_func_cgid(y_cgid_batch,f_theta_batch)
        loss_batch = loss_hgr_batch + lambda_id*loss_id_batch + lambda_cgid*loss_cgid_batch

    #### Gradient Update
    grads = tape.gradient(loss_batch,model.trainable_weights)
    optimizer.apply_gradients(zip(grads,model.trainable_weights))
    
    #### Accuracy Computations
    acc_batch_hgr = tf.keras.metrics.sparse_categorical_accuracy(y_hgr_batch,g_hgr_batch)
    acc_batch_id = tf.keras.metrics.sparse_categorical_accuracy(y_id_batch,g_id_batch)

    return loss_batch, acc_batch_hgr*y_hgr_batch.shape[0], acc_batch_id*y_id_batch.shape[0], loss_hgr_batch, loss_id_batch, loss_cgid_batch

#### Distributed Train-Step
@tf.function
def distributed_train_step(X_batch,y_batch,lambda_id,lambda_cgid,optimizer):

    """
    Function to train the model for Distributed Setup

    INPUTS:-
    1) dataset_inputs: Input to the model

    OUTPUTS:-
    1) reduced_loss_batch: Reduced Loss over the Distributed Batch
    2) reduced_acc_batch_hgr: Reduced (HGR Accuracy value for the batch) in the Distributed Batch
    3) reduced_acc_batch_id: Reduced (ID Accuracy value for the batch) in the Distributed Batch
    4) reduced_loss_hgr_batch: Reduced (HGR Loss value for the batch) in the Distributed Batch
    5) reduced_loss_id_batch: Reduced (ID Loss value for the batch) in the Distributed Batch
    6) reduced_loss_cgid_batch: Reduced CGID Loss over the Distributed Batch
    """

    #### Extracting Per-Replica Metrics
    repl_loss_batch, repl_acc_batch_hgr, repl_acc_batch_id, repl_loss_hgr_batch, repl_loss_id_batch, repl_loss_cgid_batch = strategy.run(train_step,
                                                                                                                                         args=(X_batch,y_batch,lambda_id,lambda_cgid,optimizer))
    
    #### Distributed Reduction
    reduced_loss_batch = strategy.reduce(tf.distribute.ReduceOp.SUM,repl_loss_batch,axis=None)
    reduced_acc_batch_hgr = strategy.reduce(tf.distribute.ReduceOp.SUM,repl_acc_batch_hgr,axis=None)
    reduced_acc_batch_id = strategy.reduce(tf.distribute.ReduceOp.SUM,repl_acc_batch_id,axis=None)
    reduced_loss_hgr_batch = strategy.reduce(tf.distribute.ReduceOp.SUM,repl_loss_hgr_batch,axis=None)
    reduced_loss_id_batch = strategy.reduce(tf.distribute.ReduceOp.SUM,repl_loss_id_batch,axis=None)
    reduced_loss_cgid_batch = strategy.reduce(tf.distribute.ReduceOp.SUM,repl_loss_cgid_batch,axis=None)

    return reduced_loss_batch, reduced_acc_batch_hgr, reduced_acc_batch_id, reduced_loss_hgr_batch, reduced_loss_id_batch, reduced_loss_cgid_batch

##### Test Step
#### Device Test-Step
@tf.function
def test_step(X_batch,y_batch,lambda_id,lambda_cgid):

    """
    Function to Evaluate the Model for a batch

    INPUTS:-
    1) X_batch: Batch of Tensor Arrays comprising Inputs
    2) y_batch: Batch of Tensor Arrays comprising Labels: [y_hgr,y_id,y_cgid]
    3) lambda_id: Scaling factor for ID-Loss
    4) lambda_cgid: Scaling factor for CG-ID Loss

    OUTPUTS:-
    1) loss_batch: (Loss value for the batch)*Number_of_samples_in_batch
    2) acc_batch_hgr: (HGR Accuracy value for the batch)*Number_of_samples_in_batch
    3) acc_batch_id: (ID Accuracy value for the batch)*Number_of_samples_in_batch
    """

    #### Unpacking labels
    y_hgr_batch = y_batch[:,0]
    y_id_batch = y_batch[:,1]
    y_cgid_batch = y_batch[:,2:]

    #### Output Computation
    g_hgr_batch, g_id_batch, f_theta_batch = model(X_batch)

    #### Metric Computations
    ### Loss Computations
    loss_hgr_batch = loss_func_hgr(y_hgr_batch,g_hgr_batch)
    loss_id_batch = loss_func_id(y_id_batch,g_id_batch)
    loss_cgid_batch = loss_func_cgid(y_cgid_batch,f_theta_batch)
    loss_batch = loss_hgr_batch + lambda_id*loss_id_batch + lambda_cgid*loss_cgid_batch 

    #### Accuracy Computations
    acc_batch_hgr = tf.keras.metrics.sparse_categorical_accuracy(y_hgr_batch,g_hgr_batch)
    acc_batch_id = tf.keras.metrics.sparse_categorical_accuracy(y_id_batch,g_id_batch)

    return loss_batch, acc_batch_hgr*y_hgr_batch.shape[0], acc_batch_id*y_id_batch.shape[0], loss_hgr_batch, loss_id_batch, loss_cgid_batch

#### Distributed Test-Step
@tf.function
def distributed_test_step(X_batch,y_batch,lambda_id,lambda_cgid):

    """
    Function to test the model for Distributed Setup

    INPUTS:-
    1) dataset_inputs: Input to the model

    OUTPUTS:-
    1) reduced_loss_batch: Reduced Loss over the Distributed Batch
    2) reduced_acc_batch_hgr: Reduced (HGR Accuracy value for the batch) in the Distributed Batch
    3) reduced_acc_batch_id: Reduced (ID Accuracy value for the batch) in the Distributed Batch
    4) reduced_loss_hgr_batch: Reduced (HGR Loss value for the batch) in the Distributed Batch
    5) reduced_loss_id_batch: Reduced (ID Loss value for the batch) in the Distributed Batch
    6) reduced_loss_cgid_batch: Reduced CGID Loss over the Distributed Batch
    """

    #### Extracting Per-Replica Metrics
    repl_loss_batch, repl_acc_batch_hgr, repl_acc_batch_id, repl_loss_hgr_batch, repl_loss_id_batch, repl_loss_cgid_batch = strategy.run(test_step,
                                                                                                                                         args=(X_batch,y_batch,lambda_id,lambda_cgid))
    
    #### Distributed Reduction
    reduced_loss_batch = strategy.reduce(tf.distribute.ReduceOp.SUM,repl_loss_batch,axis=None)
    reduced_acc_batch_hgr = strategy.reduce(tf.distribute.ReduceOp.SUM,repl_acc_batch_hgr,axis=None)
    reduced_acc_batch_id = strategy.reduce(tf.distribute.ReduceOp.SUM,repl_acc_batch_id,axis=None)
    reduced_loss_hgr_batch = strategy.reduce(tf.distribute.ReduceOp.SUM,repl_loss_hgr_batch,axis=None)
    reduced_loss_id_batch = strategy.reduce(tf.distribute.ReduceOp.SUM,repl_loss_id_batch,axis=None)
    reduced_loss_cgid_batch = strategy.reduce(tf.distribute.ReduceOp.SUM,repl_loss_cgid_batch,axis=None)

    return reduced_loss_batch, reduced_acc_batch_hgr, reduced_acc_batch_id, reduced_loss_hgr_batch, reduced_loss_id_batch, reduced_loss_cgid_batch

###### Training Loop
for epoch_num in range(num_epochs):
    
    print('=============================================================')
    print('Epoch Number: '+str(epoch_num+1))
    time_start = time.time() # Marking Instatiation Time
    loss_epoch = 0
    acc_epoch_hgr = 0
    acc_epoch_id = 0
    val_loss_epoch = 0
    val_acc_epoch_hgr = 0
    val_acc_epoch_id = 0
    loss_hgr_epoch = 0
    loss_id_epoch = 0
    loss_cgid_epoch = 0
    val_loss_hgr_epoch = 0
    val_loss_id_epoch = 0
    val_loss_cgid_epoch = 0

    #### Training Loop
    for batch_num, (X_batch_train,y_batch_train) in enumerate(train_dataset):
        loss_batch, acc_batch_hgr, acc_batch_id, loss_hgr_batch, loss_id_batch, loss_cgid_batch = distributed_train_step(X_batch_train,y_batch_train,lambda_id,lambda_cgid,optimizer)
        loss_epoch = loss_epoch + loss_batch # Loss for the current batch
        loss_hgr_epoch = loss_hgr_epoch + loss_hgr_batch # HGR Loss for the current batch
        loss_id_epoch = loss_id_epoch + loss_id_batch # ID Loss for the current batch
        loss_cgid_epoch = loss_cgid_epoch + loss_cgid_batch # CGID Loss for the current batch
        acc_epoch_hgr = acc_epoch_hgr + (tf.math.reduce_sum(acc_batch_hgr)/acc_batch_hgr.shape[0]) # Accuracy of the current batch
        acc_epoch_id = acc_epoch_id + (tf.math.reduce_sum(acc_batch_id)/acc_batch_id.shape[0]) # Accuracy of the current batch

    train_loss.append(loss_epoch/Total_Training_Examples)
    train_acc_hgr.append(acc_epoch_hgr/Total_Training_Examples)
    train_acc_id.append(acc_epoch_id/Total_Training_Examples)
    train_loss_collate.append([loss_hgr_epoch/Total_Training_Examples,loss_id_epoch/Total_Training_Examples,loss_cgid_epoch/(Total_Training_Examples/batch_size)])

    #### Validation Loop
    for batch_num, (X_batch_val,y_batch_val) in enumerate(val_dataset):
        val_loss_batch, val_acc_batch_hgr, val_acc_batch_id, val_loss_hgr_batch, val_loss_id_batch, val_loss_cgid_batch = distributed_test_step(X_batch_val,y_batch_val,lambda_id,lambda_cgid)
        val_loss_epoch = val_loss_epoch + val_loss_batch # Validation Loss for the current batch
        val_loss_hgr_epoch = val_loss_hgr_epoch + val_loss_hgr_batch # HGR Validation Loss for the current batch
        val_loss_id_epoch = val_loss_id_epoch + val_loss_id_batch # ID Validation Loss for the current batch
        val_loss_cgid_epoch = val_loss_cgid_epoch + val_loss_cgid_batch # CGID Validation Loss for the current batch
        val_acc_epoch_hgr = val_acc_epoch_hgr + (tf.math.reduce_sum(val_acc_batch_hgr)/val_acc_batch_hgr.shape[0]) # Accuracy of the current batch
        val_acc_epoch_id = val_acc_epoch_id + (tf.math.reduce_sum(val_acc_batch_id)/val_acc_batch_id.shape[0]) # Accuracy of the current batch

    val_loss.append(val_loss_epoch/Total_Val_Examples)
    val_acc_hgr.append(val_acc_epoch_hgr/Total_Val_Examples)
    val_acc_id.append(val_acc_epoch_id/Total_Val_Examples)
    val_loss_collate.append([val_loss_hgr_epoch/Total_Val_Examples,val_loss_id_epoch/Total_Val_Examples,val_loss_cgid_epoch/(Total_Val_Examples/batch_size)])

    #### Saving the Best Model
    if(val_loss_epoch < float(val_loss_best)):
        model.save_weights(filepath)
        val_loss_best = val_loss_epoch

    #### Displaying Metrics
    time_close = time.time() # Marking Closing Time of the loop
    print('Total Time: '+str(round((time_close - time_start), 2))+' seconds')
    print('Training Loss: '+str(float(loss_epoch/Total_Training_Examples)))
    print('Val Loss: '+str(float(val_loss_epoch/Total_Val_Examples)))
    print('Training HGR Accuracy: '+str(float(acc_epoch_hgr/Total_Training_Examples)))
    print('Training ID Accuracy: '+str(float(acc_epoch_id/Total_Training_Examples)))
    print('Val HGR Accuracy: '+str(float(val_acc_epoch_hgr/Total_Val_Examples)))
    print('Val ID Accuracy: '+str(float(val_acc_epoch_id/Total_Val_Examples)))

#history = model.fit(X_train,(y_train,y_train_id,tf.stack([y_train,y_train_id],axis=-1)),epochs=200,batch_size=32,
#                validation_data=(X_dev,(y_dev,y_dev_id,tf.stack([y_train,y_train_id],axis=-1))),validation_batch_size=32,
#                   callbacks=checkpoint)

##### Saving Training History
np.save('./Model History/'+args.exp_name+'_TrainLoss.npy',np.array(train_loss,dtype=float))
np.save('./Model History/'+args.exp_name+'_ValLoss.npy',np.array(val_loss,dtype=float))
np.save('./Model History/'+args.exp_name+'_TrainLoss-Collate.npy',np.array(train_loss_collate,dtype=float))
np.save('./Model History/'+args.exp_name+'_ValLoss-Collate.npy',np.array(val_loss_collate,dtype=float))

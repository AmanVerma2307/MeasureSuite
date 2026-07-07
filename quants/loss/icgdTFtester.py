######## Importing Libraries
import numpy as np
import tensorflow as tf

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
class ICGD_Loss(tf.keras.losses.Loss):

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
        print(y_stash.shape)

        y_hgr = y_stash[:,0] # HGR Labels - Useful for Boolean Mask Creation
        y_id = y_stash[:,3:] # Identity Labels - Useful for Disentangling Terms Estimation        

        ##### L2-Normalization
        f_theta = tf.math.l2_normalize(f_theta,axis=1)

        ##### Gramian Matrix Formation
        G_bar = tf.linalg.matmul(f_theta,f_theta,transpose_b=True)

        ##### Gramian-Matrix Positive Mask
        zero_matrix = tf.zeros_like(G_bar) # Matrix of all zeros to compare with Gramian Matrix
        Gamma_bar = tf.cast(tf.math.greater_equal(G_bar,zero_matrix),dtype=tf.float32) # Mask for all the negative values

        ##### Different Gesture Mask Computation
        delta_bar = get_negative_mask(y_hgr)
        tf.print(tf.math.reduce_sum(tf.cast(delta_bar,dtype=tf.float32)))

        ##### Lower Triangular Matrix
        LT_Mask = tf.linalg.band_part(tf.ones(shape=G_bar.shape),0,-1) # Lower Triangular Matrix

        ##### Loss Computation
        #### Defining Essentials
        Loss_CG_ID = 0 # Loss for the Current Batch
        mask_val = 0

        #### Iterating over the Identities
        for sub_idx in range(self.I):

            y_id_curr = y_id[:,sub_idx] # Extracting labels for the current identity
            delta_curr = get_positive_mask(y_id_curr) # Extracting positive mask of the current identity
            
            mask_val = mask_val + tf.math.reduce_sum(tf.cast(delta_curr,dtype=tf.float32))
            
            Loss_CG_ID_curr = tf.math.reduce_sum(tf.math.multiply(Gamma_bar,tf.math.abs(tf.math.multiply(tf.math.multiply(tf.cast(LT_Mask,dtype=tf.float32),tf.cast(delta_bar,dtype=tf.float32)),
                                                                                      tf.math.multiply(tf.cast(delta_curr,dtype=tf.float32),G_bar)))))
            Normalization_Factor = tf.math.reduce_sum(tf.math.multiply(Gamma_bar,tf.math.multiply(tf.math.multiply(tf.cast(LT_Mask,dtype=tf.float32),tf.cast(delta_bar,dtype=tf.float32)),
                                                                       tf.cast(delta_curr,dtype=tf.float32)))) 
            
            Loss_CG_ID = Loss_CG_ID + (Loss_CG_ID_curr/(Normalization_Factor+1))

        return Loss_CG_ID/self.I
    
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

f_theta = np.load('./data/soli/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz',allow_pickle=True)['arr_0']
y_hgr = np.load('./data/soli/y_dev_DeltaDistance_SOLI.npz',allow_pickle=True)['arr_0']
y_id = np.load('./data/soli/y_dev_id_DeltaDistance_SOLI.npz',allow_pickle=True)['arr_0']

y_id_ohot = get_ohot(y_id)
y_final = np.append(np.append(np.reshape(y_hgr,(y_hgr.shape[0],1)),np.reshape(y_id,(y_id.shape[0],1)),axis=-1),
                            np.append(np.reshape(y_hgr,(y_hgr.shape[0],1)),y_id_ohot,axis=-1),axis=-1)

print(y_final.shape)

icgd_loss = ICGD_Loss(32,32,10,11)
loss_val = icgd_loss(y_final,f_theta)
print(loss_val)
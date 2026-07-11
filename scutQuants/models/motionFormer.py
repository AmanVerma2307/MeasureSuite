import tensorflow as tf

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
class positionalEmbedding_mf(tf.keras.layers.Layer):
    
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
    
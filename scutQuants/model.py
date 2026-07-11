import tensorflow as tf
from models.vivit import *
from models.motionFormer import *

def getModel(args, strategy):

    if(args.modelChoice == 'vivit'):
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

    if(args.modelChoice == 'motionFormer'):
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

    if(args.modelChoice == 'mvit'):
        pass

    return model
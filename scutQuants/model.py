import tensorflow as tf
from models.vivit import *
from models.motionFormer import *
from models.mvit import *

def getModel(args, strategy):

    H = 64
    W = 64
    C_rdi = 1
    d_model = 32
    num_heads = 16
    dff_dim = 128
    p_t = 5
    p_h = 5
    p_w = 5
    rate = 0.3

    if(args.modelChoice == 'vivit'):
        T = 63
        n_h = int(((H - p_h)//p_h)+1)
        n_w = int(((W - p_w)//p_w)+1)
        n_t = int(((T - p_t)//p_t)+1)
        max_seq_len = int(n_t*(n_h/2*n_w/2))
  
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
        n_h = int(((H - p_h)//p_h)+1)
        n_w = int(((W - p_w)//p_w)+1)
        n_t = int(T/2)
        S = int(n_h/2*n_w/2)
        max_seq_len = int(n_t*(n_h/2*n_w/2))
        
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
        T = 63
        p_t = 2
        p_h = 4
        p_w = 4
        n_h = int(((H - p_h)//p_h)+1)
        n_w = int(((W - p_w)//p_w)+1)
        n_t = int(((T - p_t)//p_t)+1)
        max_seq_len = int(n_t*(n_h)*(n_w))
        rate = 0.3

        #### Res3DNet
        conv11_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
        conv12_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
        conv13_rdi = tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),padding='same',activation='relu')
        maxpool_1 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2))

        conv21_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
        conv22_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')
        conv23_rdi = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,3),padding='same',activation='relu')

        ##### ViViT

        #### tokenization
        tubelet_embedding_layer = Tubelet_Embedding(d_model,(p_t,p_h,p_w))
        positional_embedding_encoder = PositionEmbedding(max_seq_len,d_model)

        #### Stage-1
        block_11 = MViT_Encoder(d_model,d_model*2,num_heads,(2,2,2),
                                    (2,2,2),(3,3,3),(3,3,3),
                                    rate=0.3,dff_dim=128)
        block_12 = Encoder(d_model*2,num_heads,dff_dim,rate)

        #### Stage-2
        block_21 = MViT_Encoder(d_model*2,d_model*4,num_heads,(2,1,1),
                                    (2,1,1),(1,1,1),(1,1,1),
                                    rate=0.3,dff_dim=128*2)
        block_22 = Encoder(d_model*4,num_heads,dff_dim,rate)

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
            #conv13_rdi = maxpool_1(conv13_rdi)

            ### Residual Block - 2
            conv21_rdi = conv21_rdi(conv13_rdi)
            conv22_rdi = conv22_rdi(conv21_rdi)
            conv23_rdi = conv23_rdi(conv22_rdi)
            conv23_rdi = tf.keras.layers.Add()([conv23_rdi,conv21_rdi])
            #conv23_rdi = maxpool_2(conv23_rdi)

            #####  ViViT
            #### Embedding layers
            tubelet_embedding = tubelet_embedding_layer(conv23_rdi)
            tokens = positional_embedding_encoder(tubelet_embedding)

            ### Stage-1
            block_11_op, block_11_shape = block_11(tokens,[n_t,n_h,n_w])
            block_12_op = block_12(block_11_op)

            ### Stage-2
            block_21_op, block_21_shape = block_21(block_12_op,block_11_shape)
            block_22_op = block_22(block_21_op)

            ##### Output Layer
            gap_op = tf.keras.layers.GlobalAveragePooling1D()(block_22_op)
            dense1 = tf.keras.layers.Dense(32,activation='relu')(gap_op)

            #### HGR Output
            dense2_hgr = tf.keras.layers.Dense(6,activation='softmax')(dense1)

            #### ID Output
            dense2_id = tf.keras.layers.Dense(143,activation='softmax')(dense1)

            ###### Compiling Model
            model = tf.keras.models.Model(inputs=Input_Layer,outputs=[dense2_hgr,dense2_id,dense1])

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelChoice",
                        type=str,
                        default='vivit',
                        help="Number of epochs to run")
    args = parser.parse_args()

    strategy = tf.distribute.MirroredStrategy()
    model = getModel(args,strategy)
    model.summary()

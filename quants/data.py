import torch
import argparse
import numpy as np
from sklearn.utils import shuffle

def dataLoader(args):

    """
    Function to return dataLoader
    
    INPUTS:-
    1) args: Input arguments from argparse
    
    OUTPUTS:-
    1) dataLoaderTrain: The prepared training dataLoader
    2) dataLoaderTest: The prepared testing dataLoader
    """

    if(args.dataset == 'soli'):
        
        X_train = np.transpose(np.load('./_data/soli/data/X_train_DGBQA-Seen_SOLI.npz')['arr_0'],(0,4,1,2,3))
        X_dev = np.transpose(np.load('./_data/soli/data/X_dev_DGBQA-Seen_SOLI.npz')['arr_0'],(0,4,1,2,3))
        y_train = np.load('./_data/soli/data/y_train_DGBQA-Seen_SOLI.npz')['arr_0']
        y_dev = np.load('./_data/soli/data/y_dev_DGBQA-Seen_SOLI.npz')['arr_0']
        y_train_id = np.load('./_data/soli/data/y_train_id_DGBQA-Seen_SOLI.npz')['arr_0']
        y_dev_id = np.load('./_data/soli/data/y_dev_id_DGBQA-Seen_SOLI.npz')['arr_0']

    if(args.dataset == 'scut'):

        X_train = np.load('')['arr_0']
        X_dev = np.load('')['arr_0']
        y_train = np.load('')['arr_0']
        y_dev = np.load('')['arr_0']
        y_train_id = np.load('')['arr_0']
        y_dev_id = np.load('')['arr_0']

        X_train, y_train, y_train_id = shuffle(X_train,y_train,y_train_id,random_state=12)
        X_dev, y_dev, y_dev_id = shuffle(X_dev, y_dev, y_dev_id,random_state=12)
        
    if(args.dataset == 'tiny'):

        X_train = np.load('')['arr_0']
        X_dev = np.load('')['arr_0']
        y_train = np.load('')['arr_0']
        y_dev = np.load('')['arr_0']
        y_train_id = np.load('')['arr_0']
        y_dev_id = np.load('')['arr_0']

    if(args.dataset == 'handLogin'):

        X_train = np.load('')['arr_0']
        X_dev = np.load('')['arr_0']
        y_train = np.load('')['arr_0']
        y_dev = np.load('')['arr_0']
        y_train_id = np.load('')['arr_0']
        y_dev_id = np.load('')['arr_0']

    if(args.dataset == 'ntu_60'):

        X_train = np.load('./_data/ntu_rgb/dataProcessed/x_train_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        X_dev = np.load('./_data/ntu_rgb/dataProcessed/x_dev_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        y_train = np.load('./_data/ntu_rgb/dataProcessed/y_train_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        y_dev = np.load('./_data/ntu_rgb/dataProcessed/y_dev_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        y_train_id = np.load('./_data/ntu_rgb/dataProcessed/y_train_id_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        y_dev_id = np.load('./_data/ntu_rgb/dataProcessed/y_dev_id_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']

    if(args.dataset == 'ntu_120'):

        X_train = np.load('./_data/ntu_rgb/dataProcessed/x_train_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        X_dev = np.load('./_data/ntu_rgb/dataProcessed/x_dev_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        y_train = np.load('./_data/ntu_rgb/dataProcessed/y_train_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        y_dev = np.load('./_data/ntu_rgb/dataProcessed/y_dev_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        y_train_id = np.load('./_data/ntu_rgb/dataProcessed/y_train_id_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        y_dev_id = np.load('./_data/ntu_rgb/dataProcessed/y_dev_id_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        

    dataset_train = torch.utils.data.TensorDataset(torch.Tensor(X_train),
                                                   torch.Tensor(y_train),
                                                   torch.Tensor(y_train_id)
                                                   )
    dataLoader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=args.batch_size,
                                                shuffle=args.shuffle,
                                                drop_last=False)
    
    dataset_test = torch.utils.data.TensorDataset(torch.Tensor(X_dev),
                                                  torch.Tensor(y_dev),
                                                  torch.Tensor(y_dev_id)
                                                  )
    dataLoader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=args.batch_size,
                                                shuffle=args.shuffle,
                                                drop_last=False)
        
    return dataLoader_train, dataLoader_test


def dataloader_nonShuffled(args):

    """
    Function to return the Non-shuffled dataLoader
    
    INPUTS:-
    1) args: Input arguments from argparse
    
    OUTPUTS:-
    1) dataLoaderTrain: The prepared training dataLoader
    2) dataLoaderTest: The prepared testing dataLoader
    """

    if(args.dataset == 'soli'):
        
        X_dev = np.transpose(np.load('./_data/soli/X_dev_Seen-IAR-NonShuffled_SOLI.npz')['arr_0'],(0,4,1,2,3))
        y_dev = np.load('./_data/soli/y_dev_Seen-IAR-NonShuffled_SOLI.npz')['arr_0']
        y_dev_id = np.load('./_data/soli/y_dev_id_Seen-IAR-NonShuffled_SOLI.npz')['arr_0']

    if(args.dataset == 'scut'):

        X_dev = np.load('')['arr_0']
        y_dev = np.load('')['arr_0']
        y_dev_id = np.load('')['arr_0']
        
    if(args.dataset == 'tiny'):

        X_dev = np.load('')['arr_0']
        y_dev = np.load('')['arr_0']
        y_dev_id = np.load('')['arr_0']

    if(args.dataset == 'handLogin'):

        X_dev = np.load('')['arr_0']
        y_dev = np.load('')['arr_0']
        y_dev_id = np.load('')['arr_0']

    if(args.dataset == 'ntu_60'):

        X_dev = np.load('./_data/ntu_rgb/dataProcessed/x_dev_ns_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        y_dev = np.load('./_data/ntu_rgb/dataProcessed/y_dev_ns_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
        y_dev_id = np.load('./_data/ntu_rgb/dataProcessed/y_dev_id_ns_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']

    if(args.dataset == 'ntu_120'):

       X_dev = np.load('./_data/ntu_rgb/dataProcessed/x_dev_ns_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
       y_dev = np.load('./_data/ntu_rgb/dataProcessed/y_dev_ns_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
       y_dev_id = np.load('./_data/ntu_rgb/dataProcessed/y_dev_id_ns_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']


    dataset_test = torch.utils.data.TensorDataset(torch.Tensor(X_dev),
                                                  torch.Tensor(y_dev),
                                                  torch.Tensor(y_dev_id)
                                                  )
    dataLoader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=args.batch_size,
                                                shuffle=args.shuffle,
                                                drop_last=False)
        
    return dataLoader_test

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset',
                        type=str,
                        default='soli',
                        help='Dataset to be used')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='Batch size to used')
    parser.add_argument('--shuffle',
                        type=bool,
                        default=False,
                        help='Boolean to whether shuffle the datasets or not')
    
    args = parser.parse_args()

    dataLoader_train, dataLoader_test = dataLoader(args)
    device = torch.device("cuda:0")

    for batch_idx, (x,y,y_id) in enumerate(dataLoader_train):
        x = x.to(device)
        y = y.to(device)
        y_id = y_id.to(device)
        
        y_new = y + y_id
        x_new = x*x
        
        print(x.size(),y.size(),y_id.size())

    print('===========================================================')

    for batch_idx, (x,y,y_id) in enumerate(dataLoader_test):
        x = x.to(device)
        y = y.to(device)
        y_id = y_id.to(device)

        y_new = y + y_id
        x_new = x*x

        print(x.size(),y.size(),y_id.size())
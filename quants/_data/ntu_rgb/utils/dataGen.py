import os
import pickle
import argparse
import numpy as np
from sklearn.utils import shuffle
from tools import read_skeleton, makeDict, processSkeleton

parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    type=str,
                    default='ntu_60')
parser.add_argument('--numFrames',
                    type=int,
                    default=120,
                    help="Number of frames to be used")
args = parser.parse_args()

with open('./utils/reqFiles',"rb")  as file:
    reqFiles = pickle.load(file)

dataDir = './data/'
reqFiles = shuffle(reqFiles, random_state=42)

dataDict = makeDict()
actionLabels = dataDict[args.mode]['action'].keys()
idLabels = dataDict[args.mode]['id'].keys()

X_train = []
X_dev = []
y_train = []
y_dev = []
y_train_id = []
y_dev_id = []

for actionIdx, actionVal in enumerate(actionLabels): # Iteration over actions
    for subjectIdx, subjectVal in enumerate(idLabels): # Iteration over subjects

        collector = [] # List to store samples corresponding to a particular subject and actions

        for fileIdx, fileName in enumerate(reqFiles):

            if(fileName in ['S005C002P010R001A049.skeleton', 
                            'S005C002P018R001A028.skeleton',
                            'S007C003P027R002A007.skeleton',
                            'S008C003P007R001A028.skeleton',
                            'S008C003P025R002A049.skeleton',
                            'S009C002P017R001A028.skeleton',
                            'S019C002P042R001A098.skeleton',
                            'S019C002P042R001A099.skeleton',
                            'S024C002P067R001A099.skeleton',
                            'S024C002P067R001A099.skeleton',
                            'S028C003P046R001A102.skeleton',
                            'S028C003P046R001A102.skeleton',
                            'S019C002P042R001A102.skeleton',
                            'S024C002P067R001A099.skeleton',
                            'S024C002P067R001A102.skeleton',
                            'S028C003P046R002A069.skeleton',
                            'S031C002P067R001A069.skeleton',
                            'S031C003P082R002A069.skeleton'
                                ]):
                pass
            else:
                if((actionVal in fileName) and (subjectVal in fileName)):
                    collector.append(fileName)

        for itemIdx in range(len(collector)):
            if((itemIdx+1) <= int(len(collector)/2)): # Added higher samples in Training set
                itemCurr = processSkeleton(dataDir+str(collector[itemIdx]),
                                           numFrames=args.numFrames)
                X_dev.append(itemCurr)
                y_dev.append(actionIdx)
                y_dev_id.append(subjectIdx)
            else:
                itemCurr = processSkeleton(dataDir+str(collector[itemIdx]),
                                           numFrames=args.numFrames)
                X_train.append(itemCurr)
                y_train.append(actionIdx)
                y_train_id.append(subjectIdx)

        print('Processed Action: '+str(actionIdx)+' || Processed Subject:' +str(subjectIdx))

X_train = np.array(X_train)
X_dev = np.array(X_dev)
y_train = np.array(y_train)
y_dev = np.array(y_dev)
y_train_id = np.array(y_train_id)
y_dev_id = np.array(y_dev_id)

X_train, y_train, y_train_id = shuffle(X_train,
                                       y_train,
                                       y_train_id,
                                       random_state=42)

X_dev_ns = X_dev
y_dev_ns = y_dev
y_dev_id_ns = y_dev_id

X_dev, y_dev, y_dev_id = shuffle(X_dev,
                                 y_dev,
                                 y_dev_id,
                                 random_state=42)

print(X_train.shape,y_train.shape,y_train_id.shape)
print(X_dev.shape,y_dev.shape,y_dev_id.shape)

np.savez_compressed('./dataProcessed/x_train_non-idf_T'+str(args.numFrames)+'_'+str(args.mode)+'.npz',X_train)
np.savez_compressed('./dataProcessed/y_train_non-idf_T'+str(args.numFrames)+'_'+str(args.mode)+'.npz',y_train)
np.savez_compressed('./dataProcessed/y_train_id_non-idf_T'+str(args.numFrames)+'_'+str(args.mode)+'.npz',y_train_id)

np.savez_compressed('./dataProcessed/x_dev_non-idf_T'+str(args.numFrames)+'_'+str(args.mode)+'.npz',X_dev)
np.savez_compressed('./dataProcessed/y_dev_non-idf_T'+str(args.numFrames)+'_'+str(args.mode)+'.npz',y_dev)
np.savez_compressed('./dataProcessed/y_dev_id_non-idf_T'+str(args.numFrames)+'_'+str(args.mode)+'.npz',y_dev_id)

np.savez_compressed('./dataProcessed/x_dev_ns_non-idf_T'+str(args.numFrames)+'_'+str(args.mode)+'.npz',X_dev_ns)
np.savez_compressed('./dataProcessed/y_dev_ns_non-idf_T'+str(args.numFrames)+'_'+str(args.mode)+'.npz',y_dev_ns)
np.savez_compressed('./dataProcessed/y_dev_id_ns_non-idf_T'+str(args.numFrames)+'_'+str(args.mode)+'.npz',y_dev_id_ns)
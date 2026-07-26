import os
import pickle
import argparse
import numpy as np
from sklearn.utils import shuffle
from tools import read_skeleton, makeDict

parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    type=str,
                    default='ntu_60')
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
            if((actionVal in fileName) and (subjectVal in fileName)):
                collector.append(fileName)

        for itemIdx in range(len(collector)):
            if((itemIdx+1) <= int(len(collector)/2)): # Added higher samples in Training set
                itemCurr = read_skeleton(dataDir+str(collector[itemIdx]))['skel_body0']
                itemCurr = np.expand_dims(itemCurr,axis=-1)
                X_dev.append(itemCurr)
                y_dev.append(actionIdx)
                y_dev_id.append(subjectIdx)
            else:
                itemCurr = read_skeleton(dataDir+str(collector[itemIdx]))['skel_body0']
                itemCurr = np.expand_dims(itemCurr,axis=-1)
                X_train.append(itemCurr)
                y_train.append(actionIdx)
                y_train_id.append(subjectIdx)

        print('Processed Action: '+str(actionIdx)+' || Processed Subject:' +str(subjectIdx))

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

np.savez_compressed('./dataProcessed/x_train_non-idf_'+str(args.mode)+'.npz',X_train)
np.savez_compressed('./dataProcessed/y_train_non-idf_'+str(args.mode)+'.npz',y_train)
np.savez_compressed('./dataProcessed/y_train_id_non-idf_'+str(args.mode)+'.npz',y_train_id)

np.savez_compressed('./dataProcessed/x_dev_non-idf_'+str(args.mode)+'.npz',X_dev)
np.savez_compressed('./dataProcessed/y_dev_non-idf_'+str(args.mode)+'.npz',y_dev)
np.savez_compressed('./dataProcessed/y_dev_id_non-idf_'+str(args.mode)+'.npz',y_dev_id)

np.savez_compressed('./dataProcessed/x_dev_ns_non-idf_'+str(args.mode)+'.npz',X_dev_ns)
np.savez_compressed('./dataProcessed/y_dev_ns_non-idf_'+str(args.mode)+'.npz',y_dev_ns)
np.savez_compressed('./dataProcessed/y_dev_id_ns_non-idf_'+str(args.mode)+'.npz',y_dev_id_ns)

print(X_train.shape,y_train.shape,y_train_id.shape)
print(X_dev.shape,y_dev.shape,y_dev_id.shape)

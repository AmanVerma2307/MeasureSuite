import os
import torch
import pickle
import argparse
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
                X_dev.append(collector[itemIdx])
                y_dev.append(actionIdx)
                y_dev_id.append(subjectIdx)
            else:
                X_train.append(collector[itemIdx])
                y_train.append(actionIdx)
                y_train_id.append(subjectIdx)


print(X_train)
print(X_dev)
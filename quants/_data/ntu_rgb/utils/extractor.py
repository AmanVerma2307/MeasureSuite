import os
import pickle
from tools import *

# Missing file removal
dataDir = './data/'
reqFiles= [] # The required filed
dataFiles = os.listdir(dataDir)
dataFiles_filtered = [] # The filtered files with full data
missingFiles = list(load_missing_file('./utils/missingSkeletons.txt').keys())

cntrPos = 0
print('Total length: '+str(len(dataFiles)))

for fileCurr in dataFiles:
    if(fileCurr[:-9] in missingFiles):
        pass
    else:
        cntrPos = cntrPos + 1
        dataFiles_filtered.append(fileCurr)

dataFiles = dataFiles_filtered

print('Total positives: '+str(cntrPos))
print('Total length: '+str(len(dataFiles)))

# Action-class based removal
reqClass = ['A099',
            'A027',
            'A007',
            'A098',
            'A102',
            'A035',
            'A069',
            'A028',
            'A033',
            'A049']

cntrPos = 0
cntrNeg = 0

for item in os.listdir(dataDir):
    for reqItem in reqClass:
        if(reqItem in item):
            cntrPos = cntrPos + 1
            reqFiles.append(item)
            break
        else:
            pass

cntrNeg = len(dataFiles) - cntrPos

print('Total positives: '+str(cntrPos))
print('Total Negatives: '+str(cntrNeg))

with open('./utils/reqFiles',"wb") as file:
    pickle.dump(reqFiles,file)    

reqFiles_txt = open('./utils/reqFiles.txt',"w")
for idx, item in enumerate(reqFiles):
    if(idx != (len(reqFiles)-1)):
        reqFiles_txt.write(str(item)+'\n')
    else:
        reqFiles_txt.write(str(item))

import os
import pickle
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from tools import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    type=str,
                    default='analyzer')
parser.add_argument('--analyzeMode',
                    type=str,
                    default='plotSubjects')
args = parser.parse_args()

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

classMapping = {'A099':0,
                'A027':1,
                'A007':2,
                'A098':3,
                'A102':4,
                'A035':5,
                'A069':6,
                'A028':7,
                'A033':8,
                'A049':9}

if(args.mode == 'filter'):

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

if(args.mode == 'analyzer'):

    with open('./utils/reqFiles',"rb")  as file:
        reqFiles = pickle.load(file)

    if(args.analyzeMode == 'countSubjects'):
        subjectList = []
        for file in reqFiles:
            if(file[8:12] not in subjectList):
                subjectList.append(file[8:12])
        print(np.sort(subjectList))

    if(args.analyzeMode == 'plotSubjects'):
        dataMat = np.zeros(shape=(106,1))
        for file in reqFiles:
            currIdx = int(file[9:12])-1
            dataMat[currIdx,0] = dataMat[currIdx,0] + 1

        print(min(dataMat))
        print(max(dataMat))

        plt.bar(np.arange(1,107),dataMat[:,0])
        plt.show()

    if(args.analyzeMode == 'countActions'):
        dataMat = np.zeros(shape=(10,1))
        for file in reqFiles:
            currIdx = classMapping[(file[16:20])]
            dataMat[currIdx,0] = dataMat[currIdx,0] + 1

        print(min(dataMat))
        print(max(dataMat))

        plt.bar(np.arange(1,11),dataMat[:,0])
        plt.show()

    if(args.analyzeMode == 'countSubjectsActions'):
        subjectList = []
        for file in reqFiles:
            if(file[8:12] not in subjectList):
                subjectList.append(file[8:12])
        subjectList = list(np.sort(subjectList))

        def plotGramMatrix(cm,
                           cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            ax = sns.heatmap(cm, cmap=cmap, linewidth=0.5, linecolor='black')
            plt.show()

        dataMat = np.zeros(shape=(106,10))
        for file in reqFiles:
            currSubject = int(file[9:12])-1
            currAction = classMapping[(file[16:20])]
            dataMat[currSubject, currAction] = dataMat[currSubject, currAction] + 1
  
        print(dataMat)
        plotGramMatrix(dataMat >= 6)


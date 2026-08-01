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
                if(item in ['S005C002P010R001A049.skeleton']):
                    pass
                else:
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

            if(file in ['S005C002P010R001A049.skeleton', 
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
                currSubject = int(file[9:12])-1
                currAction = classMapping[(file[16:20])]
                dataMat[currSubject, currAction] = dataMat[currSubject, currAction] + 1
  
        print(dataMat)
        plotGramMatrix(dataMat >= 6)

    if(args.analyzeMode == 'countFrames'):
        frameCount = np.zeros(shape=(200,1))

        for idx, fileName in enumerate(reqFiles):
            print('idx: '+str(idx+1)+' file: '+str(fileName))
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
                itemCurr = read_skeleton('./data/'+fileName)['skel_body0']
                frames = itemCurr.shape[0]
                if(frames > (199)):
                    frames = 199
                frameCount[frames,0] = frameCount[frames,0] + 1

        print(np.mean(frameCount),np.std(frameCount),np.max(frameCount),np.min(frameCount),
              np.percentile(frameCount,q=95),
              np.percentile(frameCount,q=90),
              np.percentile(frameCount,q=80))
        plt.bar(np.arange(1,201),frameCount[:,0])
        plt.show()

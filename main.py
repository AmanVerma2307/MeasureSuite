import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.quantifiers import *
from utils.selector import get_val, select_model
from utils.retList import *
from utils.sensorSimulator import *

###### Selecting model embeddings
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    default='soli',
                    help="Dataset to be used.")
parser.add_argument("--bdbMode",
                    type=str,
                    default='Acc',
                    help="bdb sensor to be used.")
parser.add_argument("--metric",
                    type=str,
                    help="The metric to be used")
parser.add_argument('--quantifier',
                    type=str,
                    default='dgbqa',
                    help="The quantifier to be used for score generation")
parser.add_argument('--senSim_totalModels',
                    type=int,
                    default=3,
                    help="Number of models per sensor")
parser.add_argument('--initResultFile',
                    type=int,
                    default=0,
                    help="If 1, then a new result file will be instantiated.")
parser.add_argument('--nameResultFile',
                    type=str,
                    help="Name of the resultFile")
parser.add_argument('--mode',
                    type=str,
                    default='comparison',
                    help="The mode of tableGen")
parser.add_argument('--kappaVal',
                    type=float,
                    default=1.0,
                    help="Kappa parameter of nAr*")
parser.add_argument('--lambdaVal',
                    type=float,
                    default=2.0,
                    help="Lambda parameter of nAr*")
parser.add_argument('--betaVal',
                    type=float,
                    default=0.75,
                    help="Beta parameter of nAr*")
parser.add_argument('--nuVal',
                    type=float,
                    default=1.0,
                    help="nu parameter of nAr*")

args = parser.parse_args()

##### Defining essentials
if(args.dataset in ['soli','handlogin','tiny','scut','ntu_60','ntu_120']):
    embedding_list, dataset_list = retList(args.dataset)
if(args.dataset == 'bdb'):
    embedding_list, dataset_list = retList(args.dataset,
                                           bdbMode=str(args.bdbMode))

if(args.dataset == 'soli'):
    y_dev = np.load('./Embeddings/y_dev_DeltaDistance_SOLI.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DeltaDistance_SOLI.npz')['arr_0']
    G_total = 11
    I_total = 10
    labels = ['Pinch index','Palm tilt','Finger Slider','Pinch pinky','Slow Swipe','Fast Swipe','Push','Pull','Finger rub','Circle','Palm hold']
    eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]

if(args.dataset == 'handlogin'):
    y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_HandLogin.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_HandLogin.npz')['arr_0']
    G_total = 4
    I_total = 16
    labels = ['Compass','Piano','Push','Flipping fist']
    eer_values = [0.44,1.29,4.89,1.05]
   
if(args.dataset == 'tiny'):
    y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_Tiny.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_Tiny.npz')['arr_0']
    G_total = 11
    I_total = 26
    labels = ['Pinch index','Palm tilt','Finger slider','Pinch pinky','Slow swipe','Fast swipe','Push','Pull','Finger rub','Circle','Palm hold']

    e1_val = 100 - 16.45
    e2_val = 100 - 23.36 
    e1 = np.array([16.38,22.19,21.60,11.61,9.24,8.95,14.58,14.45,17.30,9.25,35.47])
    e2 = np.array([21.12,26.42,32.30,20.34,18.18,17.33,19.81,24.45,25.70,11.52,39.81])
    eer_values = (e1_val*e1+e2_val*e2)/(e1_val+e2_val)
    eer_values = list(eer_values)

if(args.dataset == 'scut'):
    y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_SCUT.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_SCUT.npz')['arr_0']
    G_total = 6
    I_total = 143
    labels = ['Fist','Rotate to Fist','Catch and Release','Four Fingers','Bend Four Fingers','Fist Opening']

    e1_val = 100 - 11.41
    e2_val = 100 - 3.293 
    e3_val = 100 - 3.659
    e1 = np.array([14.07, 13.89, 9.22, 10.84, 9.76, 10.67])
    e2 = np.array([5.511,3.667,3.044,2.26,2.489,2.778])
    e3 = np.array([3.422,5.778,3.667,3.022,3.533,2.533])
    eer_values = (e1_val*e1+e2_val*e2+e3_val*e3)/(e1_val+e2_val+e3_val)
    eer_values = list(eer_values)

if(args.dataset == 'bdb'):
    y_dev = np.load('./Embeddings/y_dev_sensor_'+args.bdbMode.lower()+'_seqLen150_bdb.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_sensor_'+args.bdbMode.lower()+'_seqLen150_bdb.npz')['arr_0']
    G_total = 4
    I_total = 51
    labels = ['keystroke','read text','gallery','tap']

    if(args.bdbMode == 'Acc'):
        e1_val = 61.80
        e2_val = 55.39
        e1 = np.array([66.23, 58.61, 62.08, 60.27])
        e2 = np.array([56.22, 52.92, 51.45, 60.98])
        eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
        eer_values = list(100 - eer_values)

    if(args.bdbMode == 'Grav'):
        e1_val = 60.61
        e2_val = 56.35
        e1 = np.array([63.84, 57.28, 60.47, 60.83])
        e2 = np.array([59.43, 56.31, 53.77, 55.88])
        eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
        eer_values = list(100 - eer_values)

    if(args.bdbMode == 'Gyro'):
        e1_val = 62.86
        e2_val = 57.80
        e1 = np.array([66.47, 59.66, 60.75, 64.56])
        e2 = np.array([58.89, 50.78, 60.53, 60.98])
        eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
        eer_values = list(100 - eer_values)

    if(args.bdbMode == 'Accl'):
        e1_val = 73.03
        e2_val = 60.06
        e1 = np.array([79.25, 64.72, 77.50, 70.66])
        e2 = np.array([67.28, 53.33, 63.73, 55.88])
        eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
        eer_values = list(100 - eer_values)

    if(args.bdbMode == 'Magn'):
        e1_val = 75.43
        e2_val = 55.60
        e1 = np.array([81.55, 72.39, 75.20, 72.58])
        e2 = np.array([60.27, 50.67, 57.36, 54.08])
        eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
        eer_values = list(100 - eer_values)

if(args.dataset== 'ntu_60'):
    y_dev = np.load('./Embeddings/y_dev_non-idf_T120_ntu_60.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_non-idf_T120_ntu_60.npz')['arr_0']
    G_total = 6
    I_total = 40
    labels = ['Jump up','Throw','Nod head','Make a phone call','Check time','Use a fan']

    eer_values = list(100 - np.array([88.96,87.25,67.03,63.11,62.22,60.37]))

if(args.dataset== 'ntu_120'):
    y_dev = np.load('./Embeddings/y_dev_non-idf_T120_ntu_120.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_non-idf_T120_ntu_120.npz')['arr_0']
    G_total = 4
    I_total = 69
    labels = ['Running on the spot','Arm swings','Side kick','Thumbs up']

    eer_values = list(100 - np.array([88.03,91.25,85.18,65.18]))

if(args.dataset in ['bdb','ntu_60','ntu_120'] and args.quantifier == 'masterFace'):
    normalize = 0
else:
    normalize = 1

if(args.mode == "comparison"):
    model = select_model(embedding_list,
                        dataset_list,
                        args.metric,
                        quantifier=args.quantifier,
                        normalize=normalize)
    print(model)

    embedding = np.load(model)['arr_0']

    val = get_val(embedding,
                y_dev,
                y_dev_id,
                eer_values,
                G_total,
                I_total,
                None,
                'full',
                quantifier=args.quantifier,
                normalize=normalize)
    
    print('nAr*: '+str(val[8]))
    print('Rank deviation: '+str(val[0]))
    print('Relevance: '+str(val[1]))
    print('Trend deviation: '+str(val[2]))
    print('Entanglement: '+str(val[3]))

    titles = ['Quantifier',
            'Metric',
            'selectedModel',
            'nAr*',
            'r',
            'R',
            'Psi',
            'Cd']
    entries = [str(args.quantifier),
            str(args.metric),
            str(model),
            str(round(val[8],4)),
            str(round(val[0],4)),
            str(round(val[1],4)),
            str(round(val[2],4)),
            str(round(val[3],4))]

    if(args.initResultFile == 1):
        resultFile = open('./_store/_resultFiles/'+args.nameResultFile+'.txt','w')

        for idx, item in enumerate(titles):
            if(idx == 2):
                resultFile.write(str(item)+'                                  ')
            if(idx == 7):
                resultFile.write(str(item)+'\n')
            if(idx in [0,1,3,4,5,6]):
                resultFile.write(str(item)+'             ')

        for idx, item in enumerate(entries):
            if(idx == 2):
                resultFile.write(str(item)+'           ')
            if(idx == 7):
                resultFile.write(str(item)+'\n')
            if(idx in [0,1,3,4,5,6]):
                resultFile.write(str(item)+'           ')

    if(args.initResultFile == 0):
        resultFile = open('./_store/_resultFiles/'+args.nameResultFile+'.txt','a')
        for idx, item in enumerate(entries):
            if(idx == 2):
                resultFile.write(str(item)+'           ')
            if(idx == 7):
                resultFile.write(str(item)+'\n')
            if(idx in [0,1,3,4,5,6]):
                resultFile.write(str(item)+'           ')

if(args.mode == 'psiComparison'):

    e_prime = 100 - np.array(eer_values)
    e_prime = (e_prime - np.mean(e_prime))/np.std(e_prime)
    e_prime = e_prime/np.linalg.norm(e_prime)
    groundTruthSorted = list(np.sort(e_prime))

    psiScores = getScores(select_model(embedding_list,
                            dataset_list,
                            'psi',
                            quantifier=args.quantifier),
                           args.quantifier,
                            y_dev,
                            y_dev_id,
                            G_total,
                            I_total)
    
    euclidScores = getScores(select_model(embedding_list,
                            dataset_list,
                            'euclid',
                            quantifier=args.quantifier),
                           args.quantifier,
                            y_dev,
                            y_dev_id,
                            G_total,
                            I_total)
    
    corrScores = getScores(select_model(embedding_list,
                            dataset_list,
                            'corr',
                            quantifier=args.quantifier),
                           args.quantifier,
                            y_dev,
                            y_dev_id,
                            G_total,
                            I_total)
    
    kendallScores = getScores(select_model(embedding_list,
                            dataset_list,
                            'Kendall',
                            quantifier=args.quantifier),
                           args.quantifier,
                            y_dev,
                            y_dev_id,
                            G_total,
                            I_total)
    
    psiScoresSorted = []
    euclidScoresSorted = []
    corrScoresSorted = []
    kendallScoresSorted = []
    labelsSorted = []

    for idx, item in enumerate(groundTruthSorted):
        for idxSort in range(G_total):
            if(item == e_prime[idxSort]):
                psiScoresSorted.append(psiScores[idxSort])
                euclidScoresSorted.append(euclidScores[idxSort])
                corrScoresSorted.append(corrScores[idxSort])
                kendallScoresSorted.append(kendallScores[idxSort])
                labelsSorted.append(labels[idxSort])
                break

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    x_axes = np.arange(start=0,stop=G_total)

    ax.plot(x_axes,
           groundTruthSorted,
           label='Ground truth',
           marker="o",
           markersize=8
           )
    ax.plot(x_axes,
           psiScoresSorted,
           label='$\\Psi$',
           marker="v",
           markersize=12
           )
    ax.plot(x_axes,
           euclidScoresSorted,
           label='Euclidean',
           marker="*",
           markersize=10,
           )
    ax.plot(x_axes,
           corrScoresSorted,
           label='$\\rho$',
           marker="h",
           markersize=8,
           )
    ax.plot(x_axes,
           kendallScoresSorted,
           label='$\\tau$',
           marker="D",
           markersize=8
           )
    
    ax.set_xticks(x_axes)
    ax.set_xticklabels(labels=labelsSorted,fontsize=10,rotation=10)
    ax.tick_params(bottom=True,left=True)

    ax.set_ylabel('Scores',fontsize=14)
    # ax.set_xlabel('Gestures',fontsize=14)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                    box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
            fancybox=True, shadow=True, ncol=5, fontsize=10)
    plt.show()
    
if(args.mode == 'stability'):

    modelSelect = select_model(embedding_list,
                                dataset_list,
                                "Ar*",
                                quantifier=args.quantifier,
                                kappaVal=args.kappaVal,
                                lambdaVal=args.lambdaVal,
                                nuVal=args.nuVal,
                                betaVal=args.betaVal,
                                normalize=normalize)
    
    labels = ['quants','kappa','lambda','nu','beta','model']
    entries = [args.quantifier,
               args.kappaVal,
               args.lambdaVal,
               args.nuVal,
               args.betaVal,
               modelSelect[13:][:-4]]

    print(entries)

    if(args.initResultFile == 1):
        resultFile = open('./_store/_stabilityFiles/'+args.nameResultFile+'.txt','w')

        for idx, item in enumerate(labels):
            if(idx in [0,1,2,3,4]):
                resultFile.write(str(item)+'      ')
            else:
                resultFile.write(str(item)+'\n')

        for idx, item in enumerate(entries):
            if(idx in [0,1,2,3,4]):
                resultFile.write(str(item)+'      ')
            else:
                resultFile.write(str(item)+'\n')

    if(args.initResultFile == 0):
        resultFile = open('./_store/_stabilityFiles/'+args.nameResultFile+'.txt','a')
        for idx, item in enumerate(entries):
            if(idx in [0,1,2,3,4]):
                resultFile.write(str(item)+'      ')
            else:
                resultFile.write(str(item)+'\n')

if(args.mode == 'sensorSimulator'):

    senSim = sensorSimulator(dataset=args.dataset,
                             quantifier=args.quantifier,
                             totalModels=args.senSim_totalModels)
    optModel = senSim.getOptModel(measure_req=args.metric,
                                  kappaVal=args.kappaVal,
                                  lambdaVal=args.lambdaVal,
                                  nuVal=args.nuVal,
                                  betaVal=args.betaVal)

    print('nAr*: '+str(optModel[0]))
    print('Rank deviation: '+str(optModel[1]))
    print('Relevance: '+str(optModel[2]))
    print('Trend deviation: '+str(optModel[3]))
    print('Entanglement: '+str(optModel[4]))

    labels = ['Quantifier',
            'Metric',
            'nAr*',
            'r',
            'R',
            'Psi',
            'Cd']
    entries = [args.quantifier,
               args.metric,
               round(optModel[0],4),
               round(optModel[1],4),
               round(optModel[2],4),
               round(optModel[3],4),
               round(optModel[4],4)]

    if(args.initResultFile == 1):
        resultFile = open('./_store/_senSimFiles/'+args.nameResultFile+'.txt','w')
        
        for idx, item in enumerate(labels):
            if(idx in [0,1,2,3,4,5]):
                resultFile.write(str(item)+'      ')
            else:
                resultFile.write(str(item)+'\n')

        for idx, item in enumerate(entries):
            if(idx in [0,1,2,3,4,5]):
                resultFile.write(str(item)+'      ')
            else:
                resultFile.write(str(item)+'\n')
        
    if(args.initResultFile == 0):
        resultFile = open('./_store/_senSimFiles/'+args.nameResultFile+'.txt','a')
        for idx, item in enumerate(entries):
            if(idx in [0,1,2,3,4,5]):
                resultFile.write(str(item)+'      ')
            else:
                resultFile.write(str(item)+'\n')
        

    
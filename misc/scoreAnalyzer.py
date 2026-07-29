import sys
sys.path.insert(1,'./')

import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.quantifiers import *
from src.AcceptanceScore import *

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',
                    type=str,
                    default='soli',
                    help="Dataset to be used.")
parser.add_argument('--exp1',
                    type=str,
                    help="First embedding to be used for analysis")
parser.add_argument('--exp2',
                    type=str,
                    help="Second embedding to be used for analysis")
parser.add_argument('--quant1',
                    type=str,
                    help="Quantifier for the first embedding")
parser.add_argument('--quant2',
                    type=str,
                    help="Qauntifier for the second embedding")
parser.add_argument('--mode',
                    type=str,
                    default="scorePlots",
                    help="Analysis to run")

args = parser.parse_args()

if(args.dataset == 'soli'):
    y_dev = np.load('./Embeddings/y_dev_DeltaDistance_SOLI.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DeltaDistance_SOLI.npz')['arr_0']
    G_total = 11
    I_total = 10
    labels = ['Pinch index','Palm tilt','Finger Slider','Pinch pinky','Slow Swipe','Fast Swipe','Push','Pull','Finger rub','Circle','Palm hold']
    eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]

if(args.dataset == 'handLogin'):
    y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_HandLogin.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_HandLogin.npz')['arr_0']
    G_total = 4
    I_total = 16
    eer_values = [0.44,1.29,4.89,1.05]

if(args.dataset == 'tiny'):
    y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_Tiny.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_Tiny.npz')['arr_0']
    G_total = 11
    I_total = 26

    e1_val = 100 - 16.45
    e2_val = 100 - 23.36 
    e1 = np.array([16.38,22.19,21.60,11.61,9.24,8.95,14.58,14.45,17.30,9.25,35.47])
    e2 = np.array([21.12,26.42,32.30,20.34,18.18,17.33,19.81,24.45,25.70,11.52,39.81])
    eer_values = (e1_val*e1+e2_val*e2)/(e1_val+e2_val)
    eer_values = list(eer_values)

score1 = getScores(embPath='./Embeddings/'+str(args.exp1)+'.npz',
                   quantifier=args.quant1,
                   y_dev=y_dev,
                   y_dev_id=y_dev_id,
                   G_total=G_total,
                   I_total=I_total)
score2 = getScores(embPath='./Embeddings/'+str(args.exp2)+'.npz',
                   quantifier=args.quant2,
                   y_dev=y_dev,
                   y_dev_id=y_dev_id,
                   G_total=G_total,
                   I_total=I_total)

e_prime = 100 - np.array(eer_values)
e_prime = (e_prime - np.mean(e_prime))/np.std(e_prime)
e_prime = e_prime/np.linalg.norm(e_prime)

ePrimeSorted = np.sort(e_prime)
labelsSorted = []
# for idxSort, item in enumerate(ePrimeSorted):
#         for idx in range(G_total)
#         if(e_prime[idx] == item)

for idx, item in enumerate(ePrimeSorted):
    for idxSort in range(G_total):
        if(item == e_prime[idxSort]):
            labelsSorted.append(labels[idxSort])
            break

def plotScores(score1,
               score2,
               groundTruth,
               G_total,
               args):
    
    """
    Function to plot scores in sorted order
    """

    groundTruthSorted = list(np.sort(groundTruth))
    score1Sorted = []
    score2Sorted = []

    for idx, item in enumerate(groundTruthSorted):
        for idxSort in range(len(groundTruthSorted)):
            if(item == groundTruth[idxSort]):
                score1Sorted.append(score1[idxSort])
                score2Sorted.append(score2[idxSort])
                break

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    x_axes = np.arange(start=0,stop=3*G_total,step=3)
    
    ax.bar(x_axes,
           groundTruthSorted,
           label='Ground truth',
           color='blue'
           )
    ax.bar(x_axes+0.8,
           score1Sorted,
           label=args.exp1,
           color='orange')
    ax.bar(x_axes+1.6,
           score1Sorted,
           label=args.exp2,
           color='tab:red')
    
    ax.set_xticks(x_axes+1.5)
    ax.set_xticklabels(labels=labelsSorted,fontsize=10,rotation=10)
    ax.tick_params(bottom=True,left=True)
    # ax.set_ylim(min(groundTruthSorted)+0.2,max(groundTruthSorted)+0.05)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                    box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=True, shadow=True, ncol=3, fontsize=10)
    plt.show()

if __name__ == "__main__":

    print(True)
    
    if(args.mode == 'scorePlots'):
        plotScores(score1,
                   score2,
                   e_prime,
                   G_total,
                   args)
        
    if(args.mode == "relVal"):

        _, relVal1 = acceptance_score(score1,
                                      e_prime,
                                      G_total,
                                      normalizer=False,
                                      relevance=False,
                                      returnRelevanceVal=True)
        print(_)
        
        _, relVal2 = acceptance_score(score2,
                                      e_prime,
                                      G_total,
                                      normalizer=False,
                                      relevance=False,
                                      returnRelevanceVal=True)
        print(_)

        print(relVal1)
        print('+++++++++++++++++++++++')
        print(relVal2)


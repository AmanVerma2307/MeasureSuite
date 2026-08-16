import sys
sys.path.insert(1,'./')

import argparse
import numpy as np
from sklearn.preprocessing import normalize as norm
from src.DGBQA_Score import gbqa_delta_dist_compute
from src.ICGDScore import CGID_Score_Calculator
from src.RankDeviation import avg_rank_deviation
from src.AcceptanceScore import acceptance_score
from src.PatternMatchDistance import pattern_match_dist
from src.AcceptanceScoreComparison import acceptance_score_comp

####### Model Arguments and Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name",
                    type=str,
                    help="Name of the Experiment being run, will be used saving the model and correponding outputs")

args = parser.parse_args()

####### Score estimation

##### Defining Essentials
gesture_list = ['Pinch index','Palm tilt','Finger slider','Pinch pinky','Slow swipe','Fast swipe','Push','Pull','Finger rub','Circle','Palm hold']
num_subjects = 26
num_gestures = 11
dgbqa_score = []
Test_Embeddings = np.load('./Embeddings/'+str(args.exp_name)+'.npz')['arr_0']
y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_Tiny.npz')['arr_0']
y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_Tiny.npz')['arr_0']

##### DGBQA Score
for g_id, gesture_curr in enumerate(gesture_list):
    print('==============================================')
    dgbqa_score_curr, d_c_star_curr, d_cs_curr, dgbqa_score_wo_curr = gbqa_delta_dist_compute(Test_Embeddings,g_id,num_subjects,y_dev,y_dev_id)
    dgbqa_score.append(dgbqa_score_curr)
    print('GBQA Delta Distance for '+str(gesture_curr)+' = '+str(dgbqa_score_curr))  

dgbqa_score = np.array(dgbqa_score) # Array Formation
dgbqa_score = (dgbqa_score - np.mean(dgbqa_score))/np.std(dgbqa_score) # Mean Normalization
dgbqa_score = dgbqa_score/np.linalg.norm(dgbqa_score) # L2-Normalization

##################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
### EER-Processing
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
##################################################################################################################

####### EER-Processing
e1_val = 100 - 16.45
e2_val = 100 - 23.36 

e1 = np.array([16.38,22.19,21.60,11.61,9.24,8.95,14.58,14.45,17.30,9.25,35.47])
e2 = np.array([21.12,26.42,32.30,20.34,18.18,17.33,19.81,24.45,25.70,11.52,39.81])

e = (e1_val*e1+e2_val*e2)/(e1_val+e2_val)
print(e)

#e = [0.44,1.29,4.89,1.05]
#e = np.array(e)
e_prime = 100 - np.array(e)
e_prime = (e_prime - np.mean(e_prime))/np.std(e_prime)
e_prime = e_prime/np.linalg.norm(e_prime)

##################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
### Feature-Space Scores
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
##################################################################################################################

####### CGID and Decorr-CGID Score
C_I, C_D = CGID_Score_Calculator(Test_Embeddings,y_dev)

##################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
### Comparison Scores    
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
##################################################################################################################

####### DGBQA-Score
###### General
beta = 0.75
alpha = 2
nu = 1
rank_dev = avg_rank_deviation(e,dgbqa_score,num_gestures)
Ar = acceptance_score(dgbqa_score,
                      e_prime,
                      num_gestures,
                      normalizer=False,
                      relevance=False)
relevance = acceptance_score(dgbqa_score,
                             e_prime,
                             num_gestures,
                             normalizer=False,
                             relevance=True)
d = pattern_match_dist(dgbqa_score,e_prime,num_gestures)
d_metric = (np.log2(2+nu*d)**(-1/alpha))
O_prime = np.exp(-beta*C_D) # Orthogonal Penalty
Ar_star = Ar*d_metric
Ar_star_plusplus = Ar_star*O_prime
Ar_max = acceptance_score(dgbqa_score,
                          e_prime,
                          num_gestures,
                          normalizer=True,
                          relevance=False)
nAr = Ar/Ar_max
nAr_star = Ar_star/Ar_max
nAr_star_plusplus = Ar_star_plusplus/Ar_max
Ar_comp = acceptance_score_comp(dgbqa_score,e_prime,num_gestures)*d_metric

print('Rank Deviation: '+str(rank_dev))
print('Relevance: '+str(relevance))
print('Ar: '+str(Ar))
print('d: '+str(d))
print('d_metric: '+str(d_metric))
print('O_prime: '+str(O_prime))
print('CGID Score: '+str(round(C_I,3)))              
print('CGID Score Decorrelated: '+str(round(C_D,3)))
print('Ar_star(Ar*d_metric): '+str(Ar_star))
print('Ar_star_++(Ar*d_metric*O_prime): '+str(Ar_star_plusplus))
print('Ar_max: '+str(Ar_max))
print('nAr: '+str(nAr))
print('nAr_star_++: '+str(nAr_star_plusplus))
print('Ar_comp: '+str(Ar_comp))

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
gesture_list = ['Compass','Piano','Push','Flipping fist']
num_subjects = 16
num_gestures = 4
dgbqa_score = []
Test_Embeddings = np.load('./Embeddings/'+str(args.exp_name)+'.npz')['arr_0']
y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_HandLogin.npz')['arr_0']
y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_HandLogin.npz')['arr_0']

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
e = [0.44,1.29,4.89,1.05]
e = np.array(e)
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

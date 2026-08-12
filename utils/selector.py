import numpy as np
import pandas as pd
from src.DGBQA_Score import gbqa_delta_dist_compute
from src.ICGDScore import CGID_Score_Calculator
from src.RankDeviation import avg_rank_deviation
from src.AcceptanceScore import acceptance_score
from src.PatternMatchDistance import pattern_match_dist
from src.quantifiers import *
from src.measures import *

####### Model selection
def get_val(embedding,
            y_dev,
            y_dev_id,
            eer_values,
            G_total,
            I_total,
            measure_req,
            mode,
            quantifier='dgbqa',
            kappaVal=1,
            lambdaVal=2,
            nuVal=1,
            betaVal=0.75,
            normalize=1):
    
    """
    Function to seek a particular measure
    """

    ##### Biometric quantification
    dgbqa_score = [] # DGBQA Score
    for g_id in range(G_total):

        if(quantifier == 'dgbqa'):
            scoreCurr, _, _, _ = gbqa_delta_dist_compute(embedding,g_id,I_total,y_dev,y_dev_id)
            dgbqa_score.append(scoreCurr)

        if(quantifier == 'deltaDistance'):
            scoreCurr = deltaDistance(embedding,
                                      g_id,
                                      I_total,
                                      y_dev,
                                      y_dev_id)
            dgbqa_score.append(scoreCurr)

        if(quantifier == 'masterFace'):
            _, d_unq, _, _ = gbqa_delta_dist_compute(embedding,g_id,I_total,y_dev,y_dev_id)
            scoreCurr = masterFace(d_unq,embedding.shape[-1])
            dgbqa_score.append(scoreCurr)

        if(quantifier == 'genCapacity'):
            scoreCurr = generativeCapacity(embedding,
                                           y_dev,
                                           y_dev_id,
                                           G_total,
                                           I_total,
                                           embedding.shape[-1],
                                           g_id)
            dgbqa_score.append(scoreCurr)

        if(quantifier == 'swipeQuality'):
            scoreCurr = swipeQuality(embedding,
                                     y_dev,
                                     G_total,
                                     g_id)
            dgbqa_score.append(scoreCurr)

    dgbqa_score = np.array(dgbqa_score) # Array Formation
    dgbqa_score = (dgbqa_score - np.mean(dgbqa_score))/np.std(dgbqa_score) # Mean Normalization
    if(normalize == 1):
        dgbqa_score = dgbqa_score/np.linalg.norm(dgbqa_score) # L2-Normalization

    ##### Ground truth equal error rates
    e_prime = 100 - np.array(eer_values)
    e_prime = (e_prime - np.mean(e_prime))/np.std(e_prime)
    if(normalize == 1):
       e_prime = e_prime/np.linalg.norm(e_prime)

    ##### Metric computation
    if(mode == 'single'): # Only one metric is required:
        if(measure_req == 'r'): # Rank deviation
            return avg_rank_deviation(np.array(eer_values),
                                    dgbqa_score,
                                    G_total)
        
        if(measure_req == 'R'): # Relevance
            return acceptance_score(dgbqa_score,
                                    e_prime,
                                    G_total,
                                    False,
                                    True)
        
        if(measure_req == 'psi'): # Pattern-match distance
            return pattern_match_dist(dgbqa_score,
                                    e_prime,
                                    G_total)
        
        if(measure_req == 'Cd'): # ICGD score
            C_I, C_D = CGID_Score_Calculator(embedding,y_dev)
            return C_D
        
        if(measure_req == 'Ar'): # Acceptance score
            return acceptance_score(dgbqa_score,
                                    e_prime,
                                    G_total,
                                    False,
                                    False,
                                    lambda_scale=lambdaVal,
                                    kappa=kappaVal)
        
        if(measure_req == 'ArCd'): # Ar*C_D
            beta = betaVal
            C_I, C_D = CGID_Score_Calculator(embedding,y_dev)
            ArCd = acceptance_score(dgbqa_score,e_prime,G_total,False,False,lambda_scale=lambdaVal,kappa=kappaVal)* np.exp(-beta*C_D)
            return ArCd
        
        if(measure_req == 'Ar_psi'): # Ar*psi
            nu = nuVal
            alpha = 2
            d = pattern_match_dist(dgbqa_score,e_prime,G_total)
            return acceptance_score(dgbqa_score,e_prime,G_total,False,False,lambda_scale=lambdaVal,kappa=kappaVal)*(np.log2(2+nu*d)**(-1/alpha))
        
        if(measure_req == 'Cd_psi'): # C_D*psi
            alpha = 2
            nu = nuVal
            beta = betaVal
            C_I, C_D = CGID_Score_Calculator(embedding,y_dev)
            d = pattern_match_dist(dgbqa_score,e_prime,G_total)
            return (np.log2(2+nu*d)**(-1/alpha))*np.exp(-beta*C_D)
        
        if(measure_req == 'Ar*'): # Full: Ar* x psi x C_D
            alpha = 2
            nu = nuVal
            beta = betaVal
            C_I, C_D = CGID_Score_Calculator(embedding,y_dev)
            d = pattern_match_dist(dgbqa_score,e_prime,G_total)
            return acceptance_score(dgbqa_score,e_prime,G_total,False,False,kappa=kappaVal,lambda_scale=lambdaVal)*(np.log2(2+nu*d)**(-1/alpha))*np.exp(-beta*C_D)
        
        if(measure_req == 'euclid'):
            return euclidean_distance(dgbqa_score,e_prime)
        
        if(measure_req == 'corr'):
            return correlation(dgbqa_score,e_prime)
        
        if(measure_req == 'DCG'):
            return compute_DCG(dgbqa_score,e_prime)
        
        if(measure_req == 'Kendall'):
            return compute_Kendalls(dgbqa_score,e_prime,G_total)
        
        if(measure_req == 'ERR'):
            return compute_ERR(dgbqa_score,e_prime,G_total)
        
        if(measure_req == 'U'):
            return compute_u(dgbqa_score,e_prime,G_total)
        
        if(measure_req == 'GRE'):
            return compute_GRE(dgbqa_score,e_prime,G_total)
        
        if(measure_req == 'infAp'):
            return compute_infAp(dgbqa_score,e_prime,G_total)
        
        if(measure_req == 'NegRel'):
            return compute_NegativeRelevance(dgbqa_score,e_prime)
        
        if(measure_req == 'RPP'):
            return compute_RPP(dgbqa_score,e_prime,G_total)

        if(measure_req == 'relEnt'):
            beta = 0.75
            C_I, C_D = CGID_Score_Calculator(embedding,y_dev)
            return acceptance_score(dgbqa_score,e_prime,G_total,False,True)*np.exp(-beta*C_D)
        
    if(mode == 'full'): # returns all nine metrics
        alpha = 2
        nu = nuVal
        beta = betaVal

        r = avg_rank_deviation(np.array(eer_values),
                                    dgbqa_score,
                                    G_total) # rank deviation
        R = acceptance_score(dgbqa_score,
                                    e_prime,
                                    G_total,
                                    False,
                                    True) # Relevance
        d = pattern_match_dist(dgbqa_score,
                                    e_prime,
                                    G_total) # Pattern match distance
        C_I, C_D = CGID_Score_Calculator(embedding,y_dev) # ICGD score
        Ar = acceptance_score(dgbqa_score,
                                    e_prime,
                                    G_total,
                                    False,
                                    False,
                                    lambda_scale=lambdaVal,
                                    kappa=kappaVal) # Ar
        ArCd = Ar* np.exp(-beta*C_D) # Ar*C_D
        Ar_psi = Ar*(np.log2(2+nu*d)**(-1/alpha)) # Ar*psi
        Cd_psi = (np.log2(2+nu*d)**(-1/alpha))*np.exp(-beta*C_D) # Cd*psi
        Ar_star = Ar*(np.log2(2+nu*d)**(-1/alpha))* np.exp(-beta*C_D) # Ar*
        Ar_star = Ar_star/(acceptance_score(dgbqa_score,e_prime,G_total,True,False,
                                            lambda_scale=lambdaVal,
                                            kappa=kappaVal))

        euclid = euclidean_distance(dgbqa_score,e_prime) # Euclidean distance
        corr = correlation(dgbqa_score,e_prime) # Correlation
        dcg = compute_DCG(dgbqa_score,e_prime) # DCG value
        kendalls_tau = compute_Kendalls(dgbqa_score,e_prime,G_total) # Kendall's Tau
        err = compute_ERR(dgbqa_score,e_prime,G_total) # ERR
        u_measure = compute_u(dgbqa_score,e_prime,G_total) # U-Measure
        gre = compute_GRE(dgbqa_score,e_prime,G_total) # GRE
        infAp = compute_infAp(dgbqa_score,e_prime,G_total) # infAp
        neg_rel = compute_NegativeRelevance(dgbqa_score,e_prime) # Negative relevance
        rpp = compute_RPP(dgbqa_score,e_prime,G_total) # RPP

        return [r,
                R,
                d,
                C_D,
                Ar,
                ArCd,
                Ar_psi,
                Cd_psi,
                Ar_star,
                euclid,
                corr,
                dcg,
                kendalls_tau,
                err,
                u_measure,
                gre,
                infAp,
                neg_rel,
                rpp] # List of measures

def get_params(embedding_list,
               dataset_list,
               var,
               quantifier='dgbqa',
               kappaVal=1,
               lambdaVal=2,
               nuVal=1,
               betaVal=0.75,
               normalize=1):

    """
    Function to get measure value for the embedding list
    
    INPUTS:-
    1) embedding_list: The list of embeddings from which the optimal is to be derived
    2) dataset_list: Corresponding list of the dataset
    3) var: 'full' or a metric
    4) quantifier: The choice of quantifier
    5) normalize: If l2 normalization is to be performed over the dgbqa scores

    OUPUTS:-
    1) measure_val: Measurment values
    """  

    measure_val = [] # Value store

    ##### Iteration over embeddings
    for idx_curr, embedding in enumerate(embedding_list):

        embedding_curr = np.load(embedding,allow_pickle=True)['arr_0']
        dataset_curr = dataset_list[idx_curr] # Current dataset

        if(dataset_curr == 'Soli'):
            y_dev = np.load('./Embeddings/y_dev_DeltaDistance_SOLI.npz')['arr_0']
            y_dev_id = np.load('./Embeddings/y_dev_id_DeltaDistance_SOLI.npz')['arr_0']
            G_total = 11
            I_total = 10
            eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]


        if(dataset_curr == 'HandLogin'):
            y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_HandLogin.npz')['arr_0']
            y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_HandLogin.npz')['arr_0']
            G_total = 4
            I_total = 16
            eer_values = [0.44,1.29,4.89,1.05]


        if(dataset_curr == 'Tiny'):
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


        if(dataset_curr == 'scut'):
            y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_SCUT.npz')['arr_0']
            y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_SCUT.npz')['arr_0']
            G_total = 6
            I_total = 143

            e1_val = 100 - 11.41
            e2_val = 100 - 3.293 
            e3_val = 100 - 3.659
            e1 = np.array([14.07, 13.89, 9.22, 10.84, 9.76, 10.67])
            e2 = np.array([5.511,3.667,3.044,2.26,2.489,2.778])
            e3 = np.array([3.422,5.778,3.667,3.022,3.533,2.533])
            eer_values = (e1_val*e1+e2_val*e2+e3_val*e3)/(e1_val+e2_val+e3_val)
            eer_values = list(eer_values)


        if(dataset_curr in ['bdbAcc','bdbGyro','bdbGrav','bdbAccl','bdbMagn']):
            y_dev = np.load('./Embeddings/y_dev_sensor_'+dataset_curr[3:].lower()+'_seqLen150_bdb.npz')['arr_0']
            y_dev_id = np.load('./Embeddings/y_dev_id_sensor_'+dataset_curr[3:].lower()+'_seqLen150_bdb.npz')['arr_0']
            G_total = 4
            I_total = 51

            if(dataset_curr == 'bdbAcc'):
                e1_val = 61.80
                e2_val = 55.39
                e1 = np.array([66.23, 58.61, 62.08, 60.27])
                e2 = np.array([56.22, 52.92, 51.45, 60.98])
                eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
                eer_values = list(100 - eer_values)

            if(dataset_curr == 'bdbGrav'):
                e1_val = 60.61
                e2_val = 56.35
                e1 = np.array([63.84, 57.28, 60.47, 60.83])
                e2 = np.array([59.43, 56.31, 53.77, 55.88])
                eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
                eer_values = list(100 - eer_values)

            if(dataset_curr == 'bdbGyro'):
                e1_val = 62.86
                e2_val = 57.80
                e1 = np.array([66.47, 59.66, 60.75, 64.56])
                e2 = np.array([58.89, 50.78, 60.53, 60.98])
                eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
                eer_values = list(100 - eer_values)

            if(dataset_curr == 'bdbAccl'):
                e1_val = 73.03
                e2_val = 60.06
                e1 = np.array([79.25, 64.72, 77.50, 70.66])
                e2 = np.array([67.28, 53.33, 63.73, 55.88])
                eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
                eer_values = list(100 - eer_values)

            if(dataset_curr == 'bdbMagn'):
                e1_val = 75.43
                e2_val = 55.60
                e1 = np.array([81.55, 72.39, 75.20, 72.58])
                e2 = np.array([60.27, 50.67, 57.36, 54.08])
                eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
                eer_values = list(100 - eer_values)
                
    
        if(dataset_curr == 'ntu_60'):
            y_dev = np.load('./Embeddings/y_dev_non-idf_T120_ntu_60.npz')['arr_0']
            y_dev_id = np.load('./Embeddings/y_dev_id_non-idf_T120_ntu_60.npz')['arr_0']
            G_total = 6
            I_total = 40

            eer_values = list(100 - np.array([88.96,87.25,67.03,63.11,62.22,60.37]))


        if(dataset_curr == 'ntu_120'):
            y_dev = np.load('./Embeddings/y_dev_non-idf_T120_ntu_120.npz')['arr_0']
            y_dev_id = np.load('./Embeddings/y_dev_id_non-idf_T120_ntu_120.npz')['arr_0']
            G_total = 4
            I_total = 69

            eer_values = list(100 - np.array([88.03,91.25,85.18,65.18]))
            
        ##### Measure computation
        if(var != 'full'):
            val_curr = get_val(embedding_curr,
                            y_dev,
                            y_dev_id,
                            eer_values,
                            G_total,
                            I_total,
                            var,
                            'single',
                            quantifier=quantifier,
                            kappaVal=kappaVal,
                            lambdaVal=lambdaVal,
                            betaVal=betaVal,
                            nuVal=nuVal,
                            normalize=normalize) # Current value
            measure_val.append(val_curr)

        if(var == 'full'):
            val_curr = get_val(embedding_curr,
                            y_dev,
                            y_dev_id,
                            eer_values,
                            G_total,
                            I_total,
                            None,
                            'full',
                            quantifier=quantifier,
                            kappaVal=kappaVal,
                            lambdaVal=lambdaVal,
                            betaVal=betaVal,
                            nuVal=nuVal,
                            normalize=normalize) # Current value
            measure_val.append(val_curr)

    return measure_val
    
def select_model(embedding_list,
                 dataset_list,
                 var,
                 quantifier,
                 kappaVal=1.0,
                 lambdaVal=2,
                 betaVal=0.75,
                 nuVal=1,
                 normalize=1
                 ):
    
    """
    Function to get optimal model as per the 'var' metric
    
    INPUTS:-
    1) embedding_list: The list of embeddings from which the optimal is to be derived
    2) dataset_list: Corresponding list of the dataset
    3) var: The measure upon which optimal is to be derived
    4) quantifier: The quantifier to be used for scoring
    5) normalize: If True, l2 normalization will be performed 

    OUPUTS:-
    1) opt_model: The optimal model/models
    """
    
    measure_val = get_params(embedding_list,
                             dataset_list,
                             var,
                             quantifier,
                             kappaVal=kappaVal,
                             lambdaVal=lambdaVal,
                             betaVal=betaVal,
                             nuVal=nuVal,
                             normalize=normalize)

    ##### Optimal selection
    if(var in ['R','Ar','ArCd','Ar_psi','Cd_psi','Ar*','corr','DCG','ERR','U','infAp','NegRel','RPP','relEnt']):
        opt_model = embedding_list[int(np.argmax(measure_val))]
    else:
        opt_model = embedding_list[int(np.argmin(measure_val))]

    return opt_model

def make_df(measure_val):
    df = pd.DataFrame()
    df['r'] = measure_val[:,0] # rank deviation
    df['relevance'] = measure_val[:,1] # Relevance
    df['psi'] = measure_val[:,2] # Pattern match distance
    df['Cd'] = measure_val[:,3] # ICGD score
    df['Ar'] = measure_val[:,4] # Acceptance score
    df['ArCd'] = measure_val[:,5] # ArCd
    df['Ar_psi'] = measure_val[:,6] # Ar_psi
    df['Cd_psi'] = measure_val[:,7] # Cd_psi
    df['Ar_star'] = measure_val[:,8] # Ar_star
    df['euclid'] = measure_val[:,9] # Euclidean distance
    df['corr'] = measure_val[:,10] # Correlation
    df['DCG'] = measure_val[:,11] # DCG
    df['Kendall'] = measure_val[:,12] # Kendall's Tau
    df['err'] = measure_val[:,13] # ERR
    df['U'] = measure_val[:,14] # U-measure
    df['gre'] = measure_val[:,15] # GRE
    df['infAp'] = measure_val[:,16] # infAp
    df['neg_rel'] = measure_val[:,17] # Negative relevance
    df['rpp'] = measure_val[:,18] # RPP
    return df

if __name__ == "__main__":

    #emebdding = np.load('./Embeddings/MS_MViT_pt5-1_SOLI.npz')['arr_0']
    y_dev = np.load('./Embeddings/y_dev_DeltaDistance_SOLI.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DeltaDistance_SOLI.npz')['arr_0']
    G_total = 11
    I_total = 10
    eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]

    #
    embedding_list = ['./Embeddings/MS_MViT_pt5-pt5_SOLI.npz',
                    './Embeddings/MS_MViT_pt5-1_SOLI.npz',
                    './Embeddings/MS_MViT_pt5-1pt5_SOLI.npz',
                    './Embeddings/MS_MViT_1-pt5_SOLI.npz',
                    './Embeddings/MS_MViT_1-1_SOLI.npz',
                    './Embeddings/MS_MViT_1-1pt5_SOLI.npz',
                    './Embeddings/MS_MViT_1pt5-pt5_SOLI.npz',
                    './Embeddings/MS_MViT_1pt5-1_SOLI.npz',
                    './Embeddings/MS_MViT_1pt5-1pt5_SOLI.npz']
    dataset_list = ['Soli']*9

    # measure_val = get_params(embedding_list,
    #                         dataset_list,
    #                         'full',
    #                         quantifier='dgbqa')
    # measure_val = np.array(measure_val)
    # print(measure_val.shape)

    # val = get_val(emebdding,y_dev,y_dev_id,eer_values,G_total,I_total,None,'full',quantifier='swipeQuality')
    # val = np.array(val)
    # print(val.shape, val)

    ##### Model selection
    opt_model = select_model(embedding_list,
                           dataset_list,
                           var='GRE',
                           quantifier='swipeQuality')
    print(opt_model)

import numpy as np
from src.DGBQA_Score import gbqa_delta_dist_compute
from src.ICGDScore import CGID_Score_Calculator
from src.AcceptanceScore import acceptance_score
from src.PatternMatchDistance import pattern_match_dist

####### Metric computation
def compute_acceptance(embeddings, 
                       y_dev, 
                       y_dev_id, 
                       eer_values, 
                       G_total,
                       I_total,
                       alpha,
                       beta,
                       lambda_scale,
                       kappa,
                       nu
                       ):
    
    """
    Function to compute normalized acceptance score
    """

    dgbqa_score = [] # DGBQA Score

    for g_id in range(G_total):
        dgbqa_score_curr, _, _, _ = gbqa_delta_dist_compute(embeddings,g_id,I_total,y_dev,y_dev_id)
        dgbqa_score.append(dgbqa_score_curr)

    dgbqa_score = np.array(dgbqa_score) # Array Formation
    dgbqa_score = (dgbqa_score - np.mean(dgbqa_score))/np.std(dgbqa_score) # Mean Normalization
    dgbqa_score = dgbqa_score/np.linalg.norm(dgbqa_score) # L2-Normalization

    e_prime = 100 - np.array(eer_values)
    e_prime = (e_prime - np.mean(e_prime))/np.std(e_prime)
    e_prime = e_prime/np.linalg.norm(e_prime)

    Ar = acceptance_score(dgbqa_score,
                          e_prime,
                          G_total,
                          False,
                          False,
                          lambda_scale,
                          kappa
                          )
    d = pattern_match_dist(dgbqa_score,e_prime,G_total)
    d_metric = (np.log2(2+nu*d)**(-1/alpha))
    C_I, C_D = CGID_Score_Calculator(embeddings,y_dev)
    O_prime = np.exp(-beta*C_D)
    Ar_star = Ar*d_metric*O_prime
    Ar_max = acceptance_score(dgbqa_score,
                              e_prime,
                              G_total,
                              True,
                              False,
                              lambda_scale,
                              kappa
                              )
    
    nAr_star = Ar_star/Ar_max
    return nAr_star

def get_nar(embedding_list,
                 y_dev_path, 
                 y_dev_id_path, 
                 eer_values, 
                 G_total,
                 I_total,
                 alpha,
                 beta,
                 lambda_scale,
                 kappa,
                 nu):
    
    """
    Function to plot surface plot wrt lambda values 
    """

    y_dev = np.load(y_dev_path)['arr_0']
    y_dev_id = np.load(y_dev_id_path)['arr_0']
    nar_value = []

    for embedding_path in embedding_list:
        embedding = np.load(embedding_path)['arr_0']
        nar_value.append(compute_acceptance(embedding,
                                            y_dev,
                                            y_dev_id,
                                            eer_values,
                                            G_total,
                                            I_total,
                                            alpha,
                                            beta,
                                            lambda_scale,
                                            kappa,
                                            nu))
    return np.array(nar_value)
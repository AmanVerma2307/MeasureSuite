######## Importing libraries                                                                                                       
import numpy as np
from Levenshtein import distance
from src.AcceptanceScore import rank_compute_acc

######## Comparison metrics

###### Standard metrics

##### Euclidean distances
def euclidean_distance(dgbqa_score,e_prime):

    """
    Euclidean distance between DGBQA scores and e_prime
    """
    return np.sqrt(np.sum(np.square(dgbqa_score-e_prime)))

##### Correlation
def correlation(dgbqa_score,e_prime):

    """
    Correlation between DGBQA scores and e_prime
    """

    return np.dot(dgbqa_score,e_prime)/(np.linalg.norm(dgbqa_score)*np.linalg.norm(e_prime))

##### DCG values
def compute_DCG(dgbqa,e_prime):
    
    """
    Function to return DCG values
    """

    #### Defining Essentials
    e_prime_sort = -(np.sort(-e_prime)) # Sorted e_prime
    dgbqa_sort = -(np.sort(-dgbqa)) # Sorted DGBQA-Score Values
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort  
    arrangement_idx = [] # List to store arrangement orders of e_prime, with the values starting from 0, and ending at G-1
    lambda_scale = 2.0
    dcg_val = 0.0

    #### Aranging DGBQA Scores as per e_prime's order
    for idx, val in enumerate(e_prime_sort):

        for idx_search, val_search in enumerate(e_prime):

            if(val == val_search):
                arrangement_idx.append(idx_search) # Finding Index on e_prime that matches with e_prime_sort
                dgbqa_re.append(dgbqa[idx_search]) # Appending terms in dgbqa_re as per the order in e_prime_sort: The best rank is the first term
                break

    ##### DCG Estimation
    for r_e_j, dgbqa_r_e_j in enumerate(dgbqa_re): # Iterating over all the gestures in the set
        Rj = 2**(lambda_scale*dgbqa_r_e_j)
        dcg_val = dcg_val + (Rj/np.log2(2+r_e_j))

    return dcg_val

##### Kendall's tau distance
def compute_Kendalls(dgbqa,e_prime,G_total):

    """
    Function to return Kendall's Tau distance values
    """

    #### String joining
    def join_str(ip,G_total):
        op = ''
        for g_idx in range(G_total):
            op = op + str(ip[g_idx])
        return op

    #### Defining Essentials
    e_prime_sort = -(np.sort(-e_prime)) # Sorted e_prime
    dgbqa_sort = -(np.sort(-dgbqa)) # Sorted DGBQA-Score Values
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort  
    arrangement_idx_e = [] # List to store arrangement orders of e_prime_sort
    arrangement_idx_d = [] # List to store arrangement orders of dgbqa_sort

    #### Arrangement index: e_prime
    for idx, val in enumerate(e_prime_sort):
        for idx_search, val_search in enumerate(e_prime):
            if(val == val_search):
                arrangement_idx_e.append(idx_search)
                dgbqa_re.append(dgbqa[idx_search])
                break
    op_e = join_str(arrangement_idx_e,G_total)

    #### Arrangement index: DGBQA
    for idx, val in enumerate(dgbqa_re): 
        for idx_search, val_search in enumerate(dgbqa_sort):
            if(val == val_search):
                arrangement_idx_d.append(idx_search)
                break
    op_d = join_str(arrangement_idx_d,G_total)
    
    return distance(op_d,op_e)

###### Standard Retrieval metrics

##### ERR
def compute_ERR(dgbqa, e_prime, G_total):

    """
    Function to compute Expected reciprocal rank
    """

    #### Defining Essentials
    e_prime_sort = -(np.sort(-e_prime)) # Sorted e_prime
    dgbqa_sort = -(np.sort(-dgbqa)) # Sorted DGBQA-Score Values
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort  
    arrangement_idx = [] # List to store arrangement orders of e_prime_sort
    err_val = 0 # ERR value

    #### Aranging DGBQA Scores as per e_prime's order
    for idx, val in enumerate(e_prime_sort):

        for idx_search, val_search in enumerate(e_prime):

            if(val == val_search):
                arrangement_idx.append(idx_search) # Finding Index on e_prime that matches with e_prime_sort
                dgbqa_re.append(dgbqa[idx_search]) # Appending terms in dgbqa_re as per the order in e_prime_sort: The best rank is the first term
                break

    dgbqa_re = (np.array(dgbqa_re)+1)/np.linalg.norm(np.array(dgbqa_re)+1) # Rescaling DGBQA
    
    ##### ERR Estimation
    for r_e_j, dgbqa_r_e_j in enumerate(dgbqa_re): # Iterating over all the gestures in the set

        err_curr = 0 # ERR value for the current rank

        if(r_e_j == 0):
            err_curr = (1/(r_e_j+1))*dgbqa_r_e_j

        if(r_e_j != 0):
            discount_curr = (1/(r_e_j+1))
            for i in range(r_e_j):
                discount_curr = discount_curr*(1-dgbqa_re[i])
            err_curr = dgbqa_r_e_j*discount_curr

        err_val = err_val + err_curr

    return err_val

##### U-Measure
def compute_u(dgbqa, e_prime, G_total):

    """
    Function to compute U-measure
    """

    #### Defining Essentials
    e_prime_sort = -(np.sort(-e_prime)) # Sorted e_prime
    dgbqa_sort = -(np.sort(-dgbqa)) # Sorted DGBQA-Score Values
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort  
    arrangement_idx = [] # List to store arrangement orders of e_prime_sort
    val = 0 

    #### Aranging DGBQA Scores as per e_prime's order
    for idx, val in enumerate(e_prime_sort):

        for idx_search, val_search in enumerate(e_prime):

            if(val == val_search):
                arrangement_idx.append(idx_search) # Finding Index on e_prime that matches with e_prime_sort
                dgbqa_re.append(dgbqa[idx_search]) # Appending terms in dgbqa_re as per the order in e_prime_sort: The best rank is the first term
                break
    
    #### ERR Estimation
    for r_e_j, dgbqa_r_e_j in enumerate(dgbqa_re): # Iterating over all the gestures in the set
        val = val + (dgbqa_r_e_j*(np.max([0,1-((r_e_j+1)/G_total)])))

    return val

##### Global rank error
def compute_GRE(dgbqa,e_prime,G):

    """
    Global rank error
    """

    ##### Defining Essentials
    e_prime_sort = -(np.sort(-e_prime)) # Sorted e_prime
    dgbqa_sort = -(np.sort(-dgbqa)) # Sorted DGBQA-Score Values
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort  
    arrangement_idx = [] # List to store arrangement orders of e_prime, with the values starting from 0, and ending at G-1
    val = 0

    #### Aranging DGBQA Scores as per e_prime's order
    for idx, val in enumerate(e_prime_sort):

        for idx_search, val_search in enumerate(e_prime):

            if(val == val_search):
                arrangement_idx.append(idx_search) # Finding Index on e_prime that matches with e_prime_sort
                dgbqa_re.append(dgbqa[idx_search]) # Appending terms in dgbqa_re as per the order in e_prime_sort: The best rank is the first term
                break
    
    #### ERR Estimation
    for r_e_j, dgbqa_r_e_j in enumerate(dgbqa_re): # Iterating over all the gestures in the set

        ### Rank Deviation Penalty
        ## Rank Computation
        rank_dgbqa = rank_compute_acc(np.array(dgbqa_sort),dgbqa_r_e_j) # Rank Derived as per DGBQA-Score Estimates
        rank_e_prime = rank_compute_acc(np.array(dgbqa_re),dgbqa_r_e_j) # Rank Derived as per e-prime-sort based sorting of DGBQA-Scores

        ## Rank Deviation
        rank_dev_j = np.abs(rank_dgbqa - rank_e_prime)

        ### Value update
        val = val + (rank_dev_j/np.log2(2+r_e_j))
    
    return val 

##### InfAP
def compute_infAp(dgbqa,e_prime,G):

    """
    Function to compute infAp
    """

    ##### Defining Essentials
    e_prime_sort = -(np.sort(-e_prime)) # Sorted e_prime
    dgbqa_sort = -(np.sort(-dgbqa)) # Sorted DGBQA-Score Values
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort  
    arrangement_idx = [] # List to store arrangement orders of e_prime, with the values starting from 0, and ending at G-1
    val = 0

    #### Aranging DGBQA Scores as per e_prime's order
    for idx, val in enumerate(e_prime_sort):

        for idx_search, val_search in enumerate(e_prime):

            if(val == val_search):
                arrangement_idx.append(idx_search) # Finding Index on e_prime that matches with e_prime_sort
                dgbqa_re.append(dgbqa[idx_search]) # Appending terms in dgbqa_re as per the order in e_prime_sort: The best rank is the first term
                break
    
    #### ERR Estimation
    for r_e_j, dgbqa_r_e_j in enumerate(dgbqa_re): # Iterating over all the gestures in the set

        relevant = 0
        val_curr = 0 

        if(r_e_j == 0):
            val_curr = 1

        if(r_e_j == G-1):

            for i in range(G-1):
                if(dgbqa_r_e_j <= dgbqa_re[i]):
                    relevant = relevant + 1

            val_curr = (1 + relevant)/G

        if(r_e_j > 0 and r_e_j != (G-1)):

            ### Left pass
            for i in range(r_e_j):
                if(dgbqa_r_e_j <= dgbqa_re[i]):
                    relevant = relevant + 1

            ### Right pass
            for i in range(r_e_j+1,G):
                if(dgbqa_r_e_j >= dgbqa_re[i]):
                    relevant = relevant + 1

            val_curr = (1/(r_e_j+1))+((r_e_j/(r_e_j+1))*(relevant/(G-1)))

        #print(val_curr)
        val = val + val_curr

    return val/G

##### Negative relevance
def compute_NegativeRelevance(dgbqa,e_prime):

    """
    Function to compute Negative Relevance
    """

    #### DCG values
    DCG = compute_DCG(dgbqa,e_prime) # Current DCG
    DCG_max = compute_DCG(e_prime,e_prime) # Maximum DCG

    #### Defining Essentials
    e_prime_sort = (np.sort(e_prime)) # Sorted e_prime
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort  
    arrangement_idx = [] # List to store arrangement orders of e_prime, with the values starting from 0, and ending at G-1

    #### Aranging DGBQA Scores as per e_prime's order
    for idx, val in enumerate(e_prime_sort):

        for idx_search, val_search in enumerate(e_prime):

            if(val == val_search):
                arrangement_idx.append(idx_search) # Finding Index on e_prime that matches with e_prime_sort
                dgbqa_re.append(dgbqa[idx_search]) # Appending terms in dgbqa_re as per the order in e_prime_sort: The best rank is the first term
                break

    #### DCG Minimum
    DCG_min = compute_DCG(np.array(dgbqa_re),np.array(e_prime))

    return (DCG - DCG_min)/(DCG_max - DCG_min)

##### RPP
def compute_RPP(dgbqa,e_prime,G):

    """
    Function to compute RPP
    """

    """
    Global rank error
    """

    ##### Defining Essentials
    e_prime_sort = -(np.sort(-e_prime)) # Sorted e_prime
    dgbqa_sort = -(np.sort(-dgbqa)) # Sorted DGBQA-Score Values
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort  
    arrangement_idx = [] # List to store arrangement orders of e_prime, with the values starting from 0, and ending at G-1
    val = 0

    #### Aranging DGBQA Scores as per e_prime's order
    for idx, val in enumerate(e_prime_sort):

        for idx_search, val_search in enumerate(e_prime):

            if(val == val_search):
                arrangement_idx.append(idx_search) # Finding Index on e_prime that matches with e_prime_sort
                dgbqa_re.append(dgbqa[idx_search]) # Appending terms in dgbqa_re as per the order in e_prime_sort: The best rank is the first term
                break
    
    #### ERR Estimation
    for r_e_j, dgbqa_r_e_j in enumerate(dgbqa_re): # Iterating over all the gestures in the set

        ### Rank Deviation Penalty
        ## Rank Computation
        rank_dgbqa = rank_compute_acc(np.array(dgbqa_sort),dgbqa_r_e_j) # Rank Derived as per DGBQA-Score Estimates
        rank_e_prime = rank_compute_acc(np.array(dgbqa_re),dgbqa_r_e_j) # Rank Derived as per e-prime-sort based sorting of DGBQA-Scores

        ## Rank Deviation
        if(rank_dgbqa == rank_e_prime):
            val = val + np.sign(dgbqa_r_e_j)*dgbqa_r_e_j
        else:
            val = val - np.sign(dgbqa_r_e_j)*(dgbqa_r_e_j)

    return val

####### Testing
#dgbqa_soli = np.array([-0.32158683, -0.09050297, -0.08070667, -0.29331375,  0.62356868,  0.3819444, -0.05457497, -0.06664492, -0.0551636,   0.33185758, -0.37487694])
#e_prime_soli = np.array([-0.36557992, -0.2823202,   0.06841959, -0.2823202,   0.34048877,  0.34638906, 0.18970344,  0.15889078,  0.12283342,  0.26771845, -0.5642232])
#G_total = 11
#err_val = correlation(dgbqa_soli,e_prime_soli)
#print(err_val)
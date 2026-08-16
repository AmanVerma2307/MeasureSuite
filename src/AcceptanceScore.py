import numpy as np

###### Rank Computation
def rank_compute_acc(rank_vector,val_to_rank):

    """
    Function to derive rank of a particular value

    INPUTS:-
    1) rank_vector: Sorted Vector in which ranking is to be performed
    2) val_to_rank: Value whoose rank is to be derived

    OUTPUTS:-
    1) rank_val: Ranked Value - The best is rank '1', but we use index of '0' all along but just the Relevance Function
    """
    #rank_vector_sort = -np.sort(-rank_vector) # Sorting the Vector: The biggest value is the better-most value

    for item_idx,item in enumerate(rank_vector): # Iterating over the vector
        if(item == val_to_rank): # Match-Found
            rank_val = item_idx # Rank Assignment 
            break

    return rank_val

####### Acceptance Score
def acceptance_score(dgbqa,
                     e_prime,
                     G,
                     normalizer, 
                     relevance, 
                     lambda_scale=2,
                     kappa=1,
                     returnRelevanceVal=False
                     ):
 
    """
    Function to compute Acceptance Score: Sum over all Gestures(Relevance/Rank Deviation)

    INPUTS:-
    1) dgbqa: Array of unranked but normalized DGBQA score values
    2) e_prime: Array of unranked but normalized (1 - EER) values
    3) G: Total number of gestures considered for analysis
    3) normalizer: If True, Ar for e_prime will be computed. Default value = False
    4) relevance: If True, relevance value is reuturned. Default value = False 

    OUTPUTS:-
    1) Ar: Acceptance Score: Sum over all Gestures(Relevance/Rank Deviation)
    """

    ##### Defining Essentials
    e_prime_sort = -(np.sort(-e_prime)) # Sorted e_prime
    dgbqa_sort = -(np.sort(-dgbqa)) # Sorted DGBQA-Score Values
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort  
    arrangement_idx = [] # List to store arrangement orders of e_prime, with the values starting from 0, and ending at G-1
    Ar = 0 # Initializing Acceptance Value as zero
    #lambda_scale = 2 # Scaling factor for relevance
    gamma = 2 # Scaling Factor for the first term in relevance formulation
    #kappa = 1  # Scaling Factor for the Rank-Deviation Penalty
    relevanceVal = []

    if(normalizer == False): # Checking if the Ar is being estimated for the normalizer or not

        ##### Aranging DGBQA Scores as per e_prime's order
        for idx, val in enumerate(e_prime_sort):

            for idx_search, val_search in enumerate(e_prime):

                if(val == val_search):
                    arrangement_idx.append(idx_search) # Finding Index on e_prime that matches with e_prime_sort
                    dgbqa_re.append(dgbqa[idx_search]) # Appending terms in dgbqa_re as per the order in e_prime_sort: The best rank is the first term
                    break

            #print(arrangement_idx)

        ##### Ar Estimation
        for r_e_j, dgbqa_r_e_j in enumerate(dgbqa_re): # Iterating over all the gestures in the set

            if(relevance == False):

                #### Relevance Gain
                R_jval = gamma*((G-(r_e_j+1)+1)/G)*dgbqa_r_e_j + ((r_e_j+1)/G)*(1 - dgbqa_r_e_j)
                R_j = 2**(lambda_scale*R_jval)

                #### Rank Deviation Penalty
                ### Rank Computation
                rank_dgbqa = rank_compute_acc(np.array(dgbqa_sort),dgbqa_r_e_j) # Rank Derived as per DGBQA-Score Estimates
                rank_e_prime = rank_compute_acc(np.array(dgbqa_re),dgbqa_r_e_j) # Rank Derived as per e-prime-sort based sorting of DGBQA-Scores

                ### Rank Deviation
                rank_dev_j = np.exp(kappa*np.abs(rank_dgbqa - rank_e_prime)) 

                #### Ar Estimate for the Current Gesture
                Ar_j = R_j/rank_dev_j
                Ar = Ar + Ar_j # Adding this to the Metric
                #print(r_e_j,R_j,np.abs(rank_dgbqa - rank_e_prime),rank_dev_j,Ar_j)
                relevanceVal.append([R_jval,np.abs(rank_dgbqa - rank_e_prime),Ar_j])

            else:

                R_j = gamma*((G-(r_e_j+1)+1)/G)*dgbqa_r_e_j + ((r_e_j+1)/G)*(1 - dgbqa_r_e_j)
                Ar = Ar + R_j # Adding this to the Metric

        if(returnRelevanceVal == True):
            return Ar, relevanceVal
        
        return Ar
     
    else:

        ##### Ar Estimation
        for r_e_j, eprime_r_e_j in enumerate(e_prime_sort): # Iterating over all the gestures in the set
            
            #### Relevance Gain
            R_j = gamma*((G - (r_e_j+1)+1)/G)*eprime_r_e_j + ((r_e_j+1)/G)*(1 - eprime_r_e_j)
            #print(r_e_j,R_j)
            R_j = 2**(lambda_scale*R_j)
            #print(r_e_j,R_j)

            #### Ar Estimate for the Current Gesture
            Ar_j = R_j
            Ar = Ar + Ar_j # Adding this to the Metric

        return Ar
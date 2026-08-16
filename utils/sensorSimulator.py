import sys
sys.path.insert(1,'./')

import numpy as np
from utils.retList import *
from src.measures import *
from src.quantifiers import *
from src.ICGDScore import CGID_Score_Calculator
from src.RankDeviation import avg_rank_deviation
from src.AcceptanceScore import acceptance_score
from src.PatternMatchDistance import pattern_match_dist


def getData(dataset):

    """
    Function to get data values
    """

    if(dataset == 'Soli'):
        y_dev = np.load('./Embeddings/y_dev_DeltaDistance_SOLI.npz')['arr_0']
        y_dev_id = np.load('./Embeddings/y_dev_id_DeltaDistance_SOLI.npz')['arr_0']
        G_total = 11
        I_total = 10
        eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]

    if(dataset == 'Tiny'):
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

    if 'bdb' in dataset:
        y_dev = np.load('./Embeddings/y_dev_sensor_'+dataset[3:].lower()+'_seqLen150_bdb.npz')['arr_0']
        y_dev_id = np.load('./Embeddings/y_dev_id_sensor_'+dataset[3:].lower()+'_seqLen150_bdb.npz')['arr_0']
        G_total = 4
        I_total = 51

        if(dataset == 'bdbAcc'):
            e1_val = 61.80
            e2_val = 55.39
            e1 = np.array([66.23, 58.61, 62.08, 60.27])
            e2 = np.array([56.22, 52.92, 51.45, 60.98])
            eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
            eer_values = list(100 - eer_values)

        if(dataset == 'bdbGrav'):
            e1_val = 60.61
            e2_val = 56.35
            e1 = np.array([63.84, 57.28, 60.47, 60.83])
            e2 = np.array([59.43, 56.31, 53.77, 55.88])
            eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
            eer_values = list(100 - eer_values)

        if(dataset == 'bdbGyro'):
            e1_val = 62.86
            e2_val = 57.80
            e1 = np.array([66.47, 59.66, 60.75, 64.56])
            e2 = np.array([58.89, 50.78, 60.53, 60.98])
            eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
            eer_values = list(100 - eer_values)

        if(dataset == 'bdbAccl'):
            e1_val = 73.03
            e2_val = 60.06
            e1 = np.array([79.25, 64.72, 77.50, 70.66])
            e2 = np.array([67.28, 53.33, 63.73, 55.88])
            eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
            eer_values = list(100 - eer_values)

        if(dataset == 'bdbMagn'):
            e1_val = 75.43
            e2_val = 55.60
            e1 = np.array([81.55, 72.39, 75.20, 72.58])
            e2 = np.array([60.27, 50.67, 57.36, 54.08])
            eer_values = (e1_val*e1 + e2_val*e2)/(e1+e2)
            eer_values = list(100 - eer_values)

    return y_dev, y_dev_id, G_total, I_total, eer_values


class sensorSimulator():

    def __init__(self,
                 dataset,
                 quantifier,
                 totalModels=3):

        self.dataset = dataset
        self.quantifier = quantifier
        self.totalModels = totalModels
        self.embeddingList, self.dataList = retList(dataset=self.dataset,
                                                    bdbMode=None,
                                                    sensorSelect=True)


        scores = np.zeros(shape=(len(self.embeddingList),len(self.embeddingList[0]))) # Initializing score values
        disentScores = np.zeros_like(scores) # Initializing disentanglement score values
        groundVal = []
        eerVal = []

        for sensorIdx in range(len(self.embeddingList)):
            for modelIdx in range(len(self.dataList[0])):

                embeddingCurr = (self.embeddingList[sensorIdx])[modelIdx]
                dataCurr = (self.dataList[sensorIdx])[modelIdx]

                y_dev, y_dev_id, G_total, I_total, groundVal_curr = getData(dataCurr)

                if(self.dataset == 'bdb' and self.quantifier == 'masterFace'):
                    normalize = 0
                else:
                    normalize = 1

                scores[sensorIdx, modelIdx] = getScores(embeddingCurr,
                                                        quantifier=self.quantifier,
                                                        y_dev=y_dev,
                                                        y_dev_id=y_dev_id,
                                                        G_total=G_total,
                                                        I_total=I_total,
                                                        normalize=normalize,
                                                        average=1)
                _, disentScores[sensorIdx, modelIdx] = CGID_Score_Calculator(np.load(embeddingCurr,
                                                                                     allow_pickle=True)['arr_0'],
                                                                                y_dev)

            eerVal.append(np.mean(groundVal_curr))

            groundVal_curr = 100 - np.array(groundVal_curr)
            groundVal.append(np.mean(groundVal_curr))

        groundVal = (groundVal - np.mean(groundVal))/np.std(groundVal) # Normalizing the collected values 
        if(normalize == 1):
            groundVal = groundVal/np.linalg.norm(groundVal)
        
        self.scores = scores
        self.disentScores = disentScores
        self.groundVal = groundVal
        self.eerVal = eerVal


        if(self.dataset == 'soliTiny'):
            totalSensors = 2
            self.dataMat = []
            self.disentMat = []
            
            for i, score1 in enumerate(self.scores[0,:]):
                for j, score2 in enumerate(self.scores[1,:]):
                    scoreVal = np.array([score1, score2])
                    if(normalize == 1):
                        scoreVal = scoreVal/np.linalg.norm(scoreVal)
                    self.dataMat.append(scoreVal)
                    

            for i, score1 in enumerate(self.disentScores[0,:]):
                for j, score2 in enumerate(self.disentScores[1,:]):
                    self.disentMat.append(np.average([score1, score2]))

        if(self.dataset == 'bdb'):
            totalSensors = 5
            self.dataMat = []
            self.disentMat = []
            
            for i, score1 in enumerate(self.disentScores[0,:]):
                for j, score2 in enumerate(self.disentScores[1,:]):
                    for k, score3 in enumerate(self.disentScores[2,:]):
                        for l, score4 in enumerate(self.disentScores[3,:]):
                            for m, score5 in enumerate(self.disentScores[4,:]):
                                scoreVal = np.array([score1, score2, score3, score4, score5])
                                scoreVal = (scoreVal - np.mean(scoreVal))/(np.std(scoreVal))
                                if(normalize == 1):
                                    scoreVal = scoreVal/np.linalg.norm(scoreVal)
                                self.dataMat.append(scoreVal)

            for i, score1 in enumerate(self.disentScores[0,:]):
                for j, score2 in enumerate(self.disentScores[1,:]):
                    for k, score3 in enumerate(self.disentScores[2,:]):
                        for l, score4 in enumerate(self.disentScores[3,:]):
                            for m, score5 in enumerate(self.disentScores[4,:]):
                                self.disentMat.append(np.average([score1, score2, score3, score4, score5]))

        self.totalSensors = totalSensors # The total number of
        self.dataMat = np.array(self.dataMat) # The matrix stores different quantification scores for all embedding combinations
        self.disentMat = np.array(self.disentMat) # The matrix stores disentanglement values for all the embedding combinations

    def getOptModel(self,
                    measure_req='Ar*',
                    lambdaVal = 2,
                    kappaVal = 1,
                    nuVal = 1,
                    betaVal = 0.75):


        """
        Function to collect measurement values and store optimal values
        """

        self.valueStore = [] # List to store values for the current measure 
        self.measureMat = [] # List to store values for the all the embeddings
        alpha = 2 

        for idx, scoresCurr in enumerate(self.dataMat):

            r = avg_rank_deviation(np.array(self.eerVal),
                                   scoresCurr,
                                   self.totalSensors)
            
            R = acceptance_score(scoresCurr,
                                 self.groundVal,
                                 self.totalSensors,
                                 False,
                                 True)
            
            d = pattern_match_dist(scoresCurr,
                                   self.groundVal,
                                   self.totalSensors)

            Cd = self.disentMat[idx]

            Ar_star = acceptance_score(scoresCurr,
                                    self.groundVal,
                                    self.totalSensors,
                                    False,
                                    False,
                                    lambda_scale=lambdaVal,
                                    kappa=kappaVal)*(np.log2(2+nuVal*d)**(-1/alpha))*np.exp(-betaVal*Cd)
            nAr_star = Ar_star/acceptance_score(scoresCurr,
                                                self.groundVal,
                                                self.totalSensors,
                                                True,
                                                False,
                                                lambda_scale=lambdaVal,
                                                kappa=kappaVal)
            
            self.measureMat.append([nAr_star,r,R,d,Cd])
            

            if(measure_req == 'r'): # Rank deviation
                self.valueStore.append(r)

            if(measure_req == 'R'): # Relevance
                self.valueStore.append(R)

            if(measure_req == 'psi'): # Trend deviation
                self.valueStore.append(d)

            if(measure_req == 'Cd'): # Disentanglement values
                self.valueStore.append(Cd)

            if(measure_req == 'Ar'): # Acceptance score
                self.valueStore.append(acceptance_score(scoresCurr,
                                                        self.groundVal,
                                                        self.totalSensors,
                                                        False,
                                                        False,
                                                        lambda_scale=lambdaVal,
                                                        kappa=kappaVal))

            if(measure_req == 'ArCd'):
                ArCd = acceptance_score(scoresCurr,
                                        self.groundVal,
                                        self.totalSensors,
                                        False,
                                        False,
                                        lambda_scale=lambdaVal,
                                        kappa=kappaVal)*np.exp(-betaVal*Cd)
                self.valueStore.append(ArCd)

            if(measure_req == 'Ar_psi'):
                Ar_psi = acceptance_score(scoresCurr,
                                        self.groundVal,
                                        self.totalSensors,
                                        False,
                                        False,
                                        lambda_scale=lambdaVal,
                                        kappa=kappaVal)*(np.log2(2+nuVal*d)**(-1/alpha))
                self.valueStore.append(Ar_psi)


            if(measure_req == 'Cd_psi'):
                self.valueStore.append((np.log2(2+nuVal*d)**(-1/alpha))*np.exp(-betaVal*Cd))

            if(measure_req == 'Ar*'):
                self.valueStore.append(nAr_star)

            if(measure_req == 'euclid'):
                self.valueStore.append(euclidean_distance(scoresCurr,
                                                          self.groundVal))

            if(measure_req == 'corr'):
                self.valueStore.append(correlation(scoresCurr,
                                                   self.groundVal))

            if(measure_req == 'DCG'):
                self.valueStore.append(compute_DCG(scoresCurr,
                                                   self.groundVal))

            if(measure_req == 'Kendall'):
                self.valueStore.append(compute_Kendalls(scoresCurr,
                                                        self.groundVal,
                                                        self.totalSensors))

            if(measure_req == 'ERR'):
                self.valueStore.append(compute_ERR(scoresCurr,
                                                   self.groundVal,
                                                   self.totalSensors))

            if(measure_req == 'U'):
                self.valueStore.append(compute_u(scoresCurr,
                                                   self.groundVal,
                                                   self.totalSensors))

            if(measure_req == 'GRE'):
                self.valueStore.append(compute_GRE(scoresCurr,
                                                 self.groundVal,
                                                 self.totalSensors))

            if(measure_req == 'infAp'):
                self.valueStore.append(compute_infAp(scoresCurr,
                                                     self.groundVal,
                                                     self.totalSensors))

            if(measure_req == 'NegRel'):
                self.valueStore.append(compute_NegativeRelevance(scoresCurr,
                                                                 self.groundVal))

            if(measure_req == 'RPP'):
                self.valueStore.append(compute_RPP(scoresCurr,
                                                   self.groundVal))

            if(measure_req == 'relEnt'):
                relEnt = acceptance_score(scoresCurr,
                                          self.groundVal,
                                          self.totalSensors,
                                          False,
                                          True)*np.exp(-betaVal*Cd)
                self.valueStore.append(relEnt)

        if(measure_req in ['R','Ar','ArCd','Ar_psi','Cd_psi','Ar*','corr','DCG','ERR','U','infAp','NegRel','RPP','relEnt']):
            optModel = self.measureMat[int(np.argmax(self.valueStore))]
        else:
            optModel = self.measureMat[int(np.argmin(self.valueStore))]

        return optModel
        
                
if __name__ == "__main__":
    senSim = sensorSimulator('bdb',quantifier='dgbqa')
    optModel = senSim.getOptModel()
    print(optModel)
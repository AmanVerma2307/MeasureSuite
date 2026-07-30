#!/bin/bash

python './correlation.py' --mode corrQuants --dataset soli --measure1 r --measure2 relevance --nameCorrFile ms_corr_soli-handlogin-tiny --initCorrFile 1
python './correlation.py' --mode corrQuants --dataset soli --measure1 r --measure2 psi --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset soli --measure1 relevance --measure2 psi --nameCorrFile ms_corr_soli-handlogin-tiny

python './correlation.py' --mode corrQuants --dataset handlogin --measure1 r --measure2 relevance --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 r --measure2 psi --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 relevance --measure2 psi --nameCorrFile ms_corr_soli-handlogin-tiny

python './correlation.py' --mode corrQuants --dataset tiny --measure1 r --measure2 relevance --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 r --measure2 psi --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 relevance --measure2 psi --nameCorrFile ms_corr_soli-handlogin-tiny




python './correlation.py' --mode corrQuants --dataset soli --measure1 r --measure2 Ar_star --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset soli --measure1 relevance --measure2 Ar_star --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset soli --measure1 psi --measure2 Ar_star --nameCorrFile ms_corr_soli-handlogin-tiny

python './correlation.py' --mode corrQuants --dataset handlogin --measure1 r --measure2 Ar_star --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 relevance --measure2 Ar_star --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 psi --measure2 Ar_star --nameCorrFile ms_corr_soli-handlogin-tiny

python './correlation.py' --mode corrQuants --dataset tiny --measure1 r --measure2 Ar_star --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 relevance --measure2 Ar_star --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 psi --measure2 Ar_star --nameCorrFile ms_corr_soli-handlogin-tiny




python './correlation.py' --mode corrQuants --dataset soli --measure1 Ar_star --measure2 euclid --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset soli --measure1 Ar_star --measure2 corr --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset soli --measure1 Ar_star --measure2 Kendall --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset soli --measure1 Ar_star --measure2 DCG --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset soli --measure1 Ar_star --measure2 err --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset soli --measure1 Ar_star --measure2 U --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset soli --measure1 Ar_star --measure2 gre --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset soli --measure1 Ar_star --measure2 infAp --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset soli --measure1 Ar_star --measure2 neg_rel --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset soli --measure1 Ar_star --measure2 rpp --nameCorrFile ms_corr_soli-handlogin-tiny

python './correlation.py' --mode corrQuants --dataset handlogin --measure1 Ar_star --measure2 euclid --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 Ar_star --measure2 corr --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 Ar_star --measure2 Kendall --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 Ar_star --measure2 DCG --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 Ar_star --measure2 err --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 Ar_star --measure2 U --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 Ar_star --measure2 gre --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 Ar_star --measure2 infAp --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 Ar_star --measure2 neg_rel --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset handlogin --measure1 Ar_star --measure2 rpp --nameCorrFile ms_corr_soli-handlogin-tiny

python './correlation.py' --mode corrQuants --dataset tiny --measure1 Ar_star --measure2 euclid --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 Ar_star --measure2 corr --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 Ar_star --measure2 Kendall --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 Ar_star --measure2 DCG --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 Ar_star --measure2 err --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 Ar_star --measure2 U --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 Ar_star --measure2 gre --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 Ar_star --measure2 infAp --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 Ar_star --measure2 neg_rel --nameCorrFile ms_corr_soli-handlogin-tiny
python './correlation.py' --mode corrQuants --dataset tiny --measure1 Ar_star --measure2 rpp --nameCorrFile ms_corr_soli-handlogin-tiny
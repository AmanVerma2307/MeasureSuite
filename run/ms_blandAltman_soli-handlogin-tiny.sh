#!/bin/bash

python './correlation.py' --mode blandAltman --dataset soli --measure1 r --measure2 relevance --baPath r-R_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 r --measure2 psi --baPath r-psi_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 relevance --measure2 psi --baPath r-R_soli

python './correlation.py' --mode blandAltman --dataset handlogin --measure1 r --measure2 relevance --baPath r-R_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 r --measure2 psi --baPath r-psi_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 relevance --measure2 psi --baPath R-psi_handlogin

python './correlation.py' --mode blandAltman --dataset tiny --measure1 r --measure2 relevance --baPath r-R_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 r --measure2 psi --baPath r-psi_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 relevance --measure2 psi --baPath R-psi_tiny




python './correlation.py' --mode blandAltman --dataset soli --measure1 r --measure2 Ar_star --baPath r-nArStar_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 relevance --measure2 Ar_star --baPath R-nArStar_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 psi --measure2 Ar_star --baPath psi-nArStar_soli

python './correlation.py' --mode blandAltman --dataset handlogin --measure1 r --measure2 Ar_star --baPath r-nArStar_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 relevance --measure2 Ar_star --baPath R-nArStar_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 psi --measure2 Ar_star --baPath psi-nArStar_handlogin

python './correlation.py' --mode blandAltman --dataset tiny --measure1 r --measure2 Ar_star --baPath r-nArStar_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 relevance --measure2 Ar_star --baPath R-nArStar_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 psi --measure2 Ar_star --baPath psi-nArStar_tiny




python './correlation.py' --mode blandAltman --dataset soli --measure1 Ar_star --measure2 euclid --baPath nArStar-euclid_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 Ar_star --measure2 corr --baPath nArStar-corr_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 Ar_star --measure2 Kendall --baPath nArStar-Kendall_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 Ar_star --measure2 DCG --baPath nArStar-DCG_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 Ar_star --measure2 err --baPath nArStar-err_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 Ar_star --measure2 U --baPath nArStar-U_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 Ar_star --measure2 gre --baPath nArStar-gre_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 Ar_star --measure2 infAp --baPath nArStar-infAp_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 Ar_star --measure2 neg_rel --baPath nArStar-negRel_soli
python './correlation.py' --mode blandAltman --dataset soli --measure1 Ar_star --measure2 rpp --baPath nArStar-rpp_soli

python './correlation.py' --mode blandAltman --dataset handlogin --measure1 Ar_star --measure2 euclid --baPath nArStar-euclid_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 Ar_star --measure2 corr --baPath nArStar-corr_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 Ar_star --measure2 Kendall --baPath nArStar-Kendall_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 Ar_star --measure2 DCG --baPath nArStar-DCG_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 Ar_star --measure2 err --baPath nArStar-err_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 Ar_star --measure2 U --baPath nArStar-U_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 Ar_star --measure2 gre --baPath nArStar-gre_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 Ar_star --measure2 infAp --baPath nArStar-infAp_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 Ar_star --measure2 neg_rel --baPath nArStar-negRel_handlogin
python './correlation.py' --mode blandAltman --dataset handlogin --measure1 Ar_star --measure2 rpp --baPath nArStar-rpp_handlogin

python './correlation.py' --mode blandAltman --dataset tiny --measure1 Ar_star --measure2 euclid --baPath nArStar-euclid_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 Ar_star --measure2 corr --baPath nArStar-corr_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 Ar_star --measure2 Kendall --baPath nArStar-Kendall_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 Ar_star --measure2 DCG --baPath nArStar-DCG_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 Ar_star --measure2 err --baPath nArStar-err_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 Ar_star --measure2 U --baPath nArStar-U_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 Ar_star --measure2 gre --baPath nArStar-gre_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 Ar_star --measure2 infAp --baPath nArStar-infAp_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 Ar_star --measure2 neg_rel --baPath nArStar-negRel_tiny
python './correlation.py' --mode blandAltman --dataset tiny --measure1 Ar_star --measure2 rpp --baPath nArStar-rpp_tiny
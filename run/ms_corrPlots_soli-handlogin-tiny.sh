#!/bin/bash

python './correlation.py' --mode corrPlots --dataset soli --measure1 r --measure2 relevance --corrPath r-R_soli
python './correlation.py' --mode corrPlots --dataset soli --measure1 r --measure2 psi --corrPath r-psi_soli
python './correlation.py' --mode corrPlots --dataset soli --measure1 relevance --measure2 psi --corrPath R-psi_soli

python './correlation.py' --mode corrPlots --dataset handlogin --measure1 r --measure2 relevance --corrPath r-R_handlogin
python './correlation.py' --mode corrPlots --dataset handlogin --measure1 r --measure2 psi --corrPath r-psi_handlogin
python './correlation.py' --mode corrPlots --dataset handlogin --measure1 relevance --measure2 psi --corrPath R-psi_handlogin

python './correlation.py' --mode corrPlots --dataset tiny --measure1 r --measure2 relevance --corrPath r-R_tiny
python './correlation.py' --mode corrPlots --dataset tiny --measure1 r --measure2 psi --corrPath r-psi_tiny
python './correlation.py' --mode corrPlots --dataset tiny --measure1 relevance --measure2 psi --corrPath R-psi_tiny


python './correlation.py' --mode corrPlots --dataset soli --measure1 r --measure2 Ar_star --corrPath r-nArStar_soli
python './correlation.py' --mode corrPlots --dataset soli --measure1 relevance --measure2 Ar_star --corrPath R-nArStar_soli
python './correlation.py' --mode corrPlots --dataset soli --measure1 psi --measure2 Ar_star --corrPath psi-nArStar_soli

python './correlation.py' --mode corrPlots --dataset handlogin --measure1 r --measure2 Ar_star --corrPath r-nArStar_handlogin
python './correlation.py' --mode corrPlots --dataset handlogin --measure1 relevance --measure2 Ar_star --corrPath R-nArStar_handlogin
python './correlation.py' --mode corrPlots --dataset handlogin --measure1 psi --measure2 Ar_star --corrPath psi-nArStar_handlogin

python './correlation.py' --mode corrPlots --dataset tiny --measure1 r --measure2 Ar_star --corrPath r-nArStar_tiny
python './correlation.py' --mode corrPlots --dataset tiny --measure1 relevance --measure2 Ar_star --corrPath r-nArStar_tiny
python './correlation.py' --mode corrPlots --dataset tiny --measure1 psi --measure2 Ar_star --corrPath psi-nArStar_tiny
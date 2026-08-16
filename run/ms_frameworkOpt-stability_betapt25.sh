#!/bin/bash

python './tableGen.py' --metric 'Ar' --quantifier 'dgbqa' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25' --initResultFile 1
python './tableGen.py' --metric 'Ar' --quantifier 'deltaDistance' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'
python './tableGen.py' --metric 'Ar' --quantifier 'masterFace' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'
python './tableGen.py' --metric 'Ar' --quantifier 'genCapacity' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'
python './tableGen.py' --metric 'Ar' --quantifier 'swipeQuality' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'

python './tableGen.py' --dataset handLogin --metric 'Ar' --quantifier 'dgbqa' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'
python './tableGen.py' --dataset handLogin --metric 'Ar' --quantifier 'deltaDistance' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'
python './tableGen.py' --dataset handLogin --metric 'Ar' --quantifier 'masterFace' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'
python './tableGen.py' --dataset handLogin --metric 'Ar' --quantifier 'genCapacity' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'
python './tableGen.py' --dataset handLogin --metric 'Ar' --quantifier 'swipeQuality' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'

python './tableGen.py' --dataset tiny --metric 'Ar' --quantifier 'dgbqa' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'
python './tableGen.py' --dataset tiny --metric 'Ar' --quantifier 'deltaDistance' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'
python './tableGen.py' --dataset tiny --metric 'Ar' --quantifier 'masterFace' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'
python './tableGen.py' --dataset tiny --metric 'Ar' --quantifier 'genCapacity' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'
python './tableGen.py' --dataset tiny --metric 'Ar' --quantifier 'swipeQuality' --nameResultFile 'ms_frameworkOpt-stability_Ar-betapt1-kappapt25'

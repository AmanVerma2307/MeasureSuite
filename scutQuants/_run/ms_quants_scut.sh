#!/bin/bash

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 0.5 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_mvit_pt5-pt01_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 0.5 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_mvit_pt5-pt01_scut"

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 0.5 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_mvit_pt5-pt05_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 0.5 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_mvit_pt5-pt05_scut"

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 0.5 --lambda_cgid 0.5 --local_batch_size 8 --exp_name "ms_mvit_pt5-pt5_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 0.5 --lambda_cgid 0.5 --local_batch_size 8 --exp_name "ms_mvit_pt5-pt5_scut"

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 0.5 --lambda_cgid 1.0 --local_batch_size 8 --exp_name "ms_mvit_pt5-1_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 0.5 --lambda_cgid 1.0 --local_batch_size 8 --exp_name "ms_mvit_pt5-1_scut"

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 0.5 --lambda_cgid 1.5 --local_batch_size 8 --exp_name "ms_mvit_pt5-1pt5_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 0.5 --lambda_cgid 1.5 --local_batch_size 8 --exp_name "ms_mvit_pt5-1pt5_scut"


python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 1.0 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_mvit_1-pt01_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 1.0 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_mvit_1-pt01_scut"

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 1.0 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_mvit_1-pt05_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 1.0 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_mvit_1-pt05_scut"

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 1.0 --lambda_cgid 0.5 --local_batch_size 8 --exp_name "ms_mvit_1-pt5_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 1.0 --lambda_cgid 0.5 --local_batch_size 8 --exp_name "ms_mvit_1-pt5_scut"

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 1.0 --lambda_cgid 1.0 --local_batch_size 8 --exp_name "ms_mvit_1-1_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 1.0 --lambda_cgid 1.0 --local_batch_size 8 --exp_name "ms_mvit_1-1_scut"

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 1.0 --lambda_cgid 1.5 --local_batch_size 8 --exp_name "ms_mvit_1-1pt5_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 1.0 --lambda_cgid 1.5 --local_batch_size 8 --exp_name "ms_mvit_1-1pt5_scut"


python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 1.5 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_mvit_1pt5-pt01_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 1.5 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_mvit_1pt5-pt01_scut"

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 1.5 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_mvit_1pt5-pt05_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 1.5 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_mvit_1pt5-pt05_scut"

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 1.5 --lambda_cgid 0.5 --local_batch_size 8 --exp_name "ms_mvit_1pt5-pt5_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 1.5 --lambda_cgid 0.5 --local_batch_size 8 --exp_name "ms_mvit_1pt5-pt5_scut"

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 1.5 --lambda_cgid 1.0 --local_batch_size 8 --exp_name "ms_mvit_1pt5-1_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 1.5 --lambda_cgid 1.0 --local_batch_size 8 --exp_name "ms_mvit_1pt5-1_scut"

python './scutQuants/trainer.py' --modelChoice mvit --lambda_id 1.5 --lambda_cgid 1.5 --local_batch_size 8 --exp_name "ms_mvit_1pt5-1pt5_scut"
python './scutQuants/tester.py' --modelChoice mvit --lambda_id 1.5 --lambda_cgid 1.5 --local_batch_size 8 --exp_name "ms_mvit_1pt5-1pt5_scut"




python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 0.5 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_mf_pt5-pt01_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 0.5 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_mf_pt5-pt01_scut"

python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 0.5 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_mf_pt5-pt05_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 0.5 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_mf_pt5-pt05_scut"

python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 0.5 --lambda_cgid 0.5 --local_batch_size 8 --exp_name "ms_mf_pt5-pt5_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 0.5 --lambda_cgid 0.5 --local_batch_size 8 --exp_name "ms_mf_pt5-pt5_scut"

python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 0.5 --lambda_cgid 1.0 --local_batch_size 8 --exp_name "ms_mf_pt5-1_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 0.5 --lambda_cgid 1.0 --local_batch_size 8 --exp_name "ms_mf_pt5-1_scut"

python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 0.5 --lambda_cgid 1.5 --local_batch_size 8 --exp_name "ms_mf_pt5-1pt5_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 0.5 --lambda_cgid 1.5 --local_batch_size 8 --exp_name "ms_mf_pt5-1pt5_scut"


python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 1.0 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_mf_1-pt01_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 1.0 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_mf_1-pt01_scut"

python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 1.0 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_mf_1-pt05_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 1.0 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_mf_1-pt05_scut"

python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 1.0 --lambda_cgid 0.5 --local_batch_size 8 --exp_name "ms_mf_1-pt5_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 1.0 --lambda_cgid 0.5 --local_batch_size 8 --exp_name "ms_mf_1-pt5_scut"

python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 1.0 --lambda_cgid 1.0 --local_batch_size 8 --exp_name "ms_mf_1-1_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 1.0 --lambda_cgid 1.0 --local_batch_size 8 --exp_name "ms_mf_1-1_scut"

python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 1.0 --lambda_cgid 1.5 --local_batch_size 8 --exp_name "ms_mf_1-1pt5_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 1.0 --lambda_cgid 1.5 --local_batch_size 8 --exp_name "ms_mf_1-1pt5_scut"


python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 1.5 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_mf_1pt5-pt01_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 1.5 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_mf_1pt5-pt01_scut"

python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 1.5 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_mf_1pt5-pt05_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 1.5 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_mf_1pt5-pt05_scut"

python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 1.5 --lambda_cgid 0.5 --local_batch_size 8 --exp_name "ms_mf_1pt5-pt5_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 1.5 --lambda_cgid 0.5 --local_batch_size 8 --exp_name "ms_mf_1pt5-pt5_scut"

python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 1.5 --lambda_cgid 1.0 --local_batch_size 8 --exp_name "ms_mf_1pt5-1_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 1.5 --lambda_cgid 1.0 --local_batch_size 8 --exp_name "ms_mf_1pt5-1_scut"

python './scutQuants/trainer.py' --modelChoice motionFormer --lambda_id 1.5 --lambda_cgid 1.5 --local_batch_size 8 --exp_name "ms_mf_1pt5-1pt5_scut"
python './scutQuants/tester.py' --modelChoice motionFormer --lambda_id 1.5 --lambda_cgid 1.5 --local_batch_size 8 --exp_name "ms_mf_1pt5-1pt5_scut"




python './scutQuants/trainer.py' --modelChoice vivit --lambda_id 0.5 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_vivit_pt5-pt01_scut"
python './scutQuants/tester.py' --modelChoice vivit --lambda_id 0.5 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_vivit_pt5-pt01_scut"

python './scutQuants/trainer.py' --modelChoice vivit --lambda_id 0.5 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_vivit_pt5-pt05_scut"
python './scutQuants/tester.py' --modelChoice vivit --lambda_id 0.5 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_vivit_pt5-pt05_scut"


python './scutQuants/trainer.py' --modelChoice vivit --lambda_id 1.0 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_vivit_1-pt01_scut"
python './scutQuants/tester.py' --modelChoice vivit --lambda_id 1.0 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_vivit_1-pt01_scut"

python './scutQuants/trainer.py' --modelChoice vivit --lambda_id 1.0 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_vivit_1-pt05_scut"
python './scutQuants/tester.py' --modelChoice vivit --lambda_id 1.0 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_vivit_1-pt05_scut"


python './scutQuants/trainer.py' --modelChoice vivit --lambda_id 1.5 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_vivit_1pt5-pt01_scut"
python './scutQuants/tester.py' --modelChoice vivit --lambda_id 1.5 --lambda_cgid 0.01 --local_batch_size 8 --exp_name "ms_vivit_1pt5-pt01_scut"

python './scutQuants/trainer.py' --modelChoice vivit --lambda_id 1.5 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_vivit_1pt5-pt05_scut"
python './scutQuants/tester.py' --modelChoice vivit --lambda_id 1.5 --lambda_cgid 0.05 --local_batch_size 8 --exp_name "ms_vivit_1pt5-pt05_scut"
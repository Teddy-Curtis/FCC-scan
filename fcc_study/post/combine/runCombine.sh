#!/bin/bash
combine_direc=$1
mH=$2
mA=$3

echo Running for
echo mH = $mH
echo mA = $mA
echo combine_direc = $combine_direc


ulimit -s unlimited
source ~/.bashrc
cd //vols/cms/emc21/idmStudy/CMSSW_14_0_0_pre0/src
cmsenv

cd ${combine_direc}

combineCards.py MuMu=MuMu_datacard.txt EE=EE_datacard.txt  > combined_datacard.txt

combine -M AsymptoticLimits MuMu_datacard.txt -m ${mH} --keyword-value MA=${mA} -n MuMu
python3 /vols/cms/emc21/FCC/FCC-Study/fcc_study/post/combine/getLimitFromFile.py --combine_direc ${combine_direc} --extra_name MuMu --object_name limits

combine -M AsymptoticLimits EE_datacard.txt -m ${mH} --keyword-value MA=${mA} -n EE
python3 /vols/cms/emc21/FCC/FCC-Study/fcc_study/post/combine/getLimitFromFile.py --combine_direc ${combine_direc} --extra_name EE --object_name limits

combine -M AsymptoticLimits combined_datacard.txt -m ${mH} --keyword-value MA=${mA} -n combined
python3 /vols/cms/emc21/FCC/FCC-Study/fcc_study/post/combine/getLimitFromFile.py --combine_direc ${combine_direc} --extra_name combined --object_name limits

combine -M Significance MuMu_datacard.txt -m ${mH} --keyword-value MA=${mA} -t -1 --expectSignal=1 -n MuMu
python3 /vols/cms/emc21/FCC/FCC-Study/fcc_study/post/combine/getLimitFromFile.py --combine_direc ${combine_direc} --extra_name MuMu --object_name significance

combine -M Significance EE_datacard.txt -m ${mH} --keyword-value MA=${mA} -t -1 --expectSignal=1 -n EE
python3 /vols/cms/emc21/FCC/FCC-Study/fcc_study/post/combine/getLimitFromFile.py --combine_direc ${combine_direc} --extra_name EE --object_name significance

combine -M Significance combined_datacard.txt -m ${mH} --keyword-value MA=${mA} -t -1 --expectSignal=1 -n combined
python3 /vols/cms/emc21/FCC/FCC-Study/fcc_study/post/combine/getLimitFromFile.py --combine_direc ${combine_direc} --extra_name combined --object_name significance



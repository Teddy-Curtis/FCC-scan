#!/bin/bash


echo "[wrapper] hostname  = " `hostname`
echo "[wrapper] date      = " `date`
echo "[wrapper] linux timestamp = " `date +%s`

echo "[wrapper] ls-ing files"
ls -altrh

eval "$(micromamba shell hook --shell bash)"
micromamba activate FCC-forAMstudent2

cd /vols/cms/emc21/FCC/FCC-Study

python main.py

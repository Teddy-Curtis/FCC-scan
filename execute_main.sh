#!/bin/bash


echo "[wrapper] hostname  = " `hostname`
echo "[wrapper] date      = " `date`
echo "[wrapper] linux timestamp = " `date +%s`

echo "[wrapper] ls-ing files"
ls -altrh

eval "$(micromamba shell hook --shell bash)"
micromamba activate fcc-study

cd /vols/cms/emc21/fccStudy

python main.py

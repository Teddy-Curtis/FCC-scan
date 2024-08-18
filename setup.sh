#!/bin/bash
pip install mplhep
conda install pytorch==1.12.1 cudatoolkit=10.2 -c pytorch -c nvidia

scp -r ecurtis@lxplus.cern.ch://eos/user/a/amagnan/FCC/iDMprod/winter2023/stage3 data

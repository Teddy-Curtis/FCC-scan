# FCC Study

Repo for FCC study analysis. This centres around a parametric neural network 
that is used for signal extraction. 

**1. Clone this repository**  

```
git clone git@github.com:Teddy-Curtis/FCC-scan.git
```

**2. Install dependencies**  

All the dependencies are stored in ```environment.yml``` and can be installed 
using conda, mamba or micromamba. NOTE: If you are using conda, and it is taking 
forever to install the environmenrt, then follow the instructions in the link:
https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community . 

I used conda, but if you want to install with mamba then just replace 
```conda``` with ```micromamba```. Note: You will also have to do the same 
swap in the ```setup.sh``` file as well.

To install with conda:
```
# First clean conda, press 'y' for each option to delete the stored tarballs etc.
conda clean -a
# Now create the environment
conda env create -f environment.yml
```

If you are having problems with storage during the conda env create bit, then 
create a directory in your EOS folder, called e.g. NEWTMP, then set that as 
the new temp directory before running ```conda env create```:
```
mkdir YOUR_EOS_DIRECTORY/NEWTMP
export TMPDIR="YOUR_EOS_DIRECTORY/NEWTMP";  conda env create -f environment.yml
```

If this has now worked fine, then you can move onto the following:
Following this run:
```
conda activate fcc-study
```


Next run ```setup.sh``` which will install mplhep, pytorch and also copyt all of
the data files. 

NOTE: For pytorch GPU to be installed correctly (and not the cpu variation) 
you need to run ```./setup.sh``` on a system with a GPU. For Imperial that means
first ssh'ing to the gpu00 node adn then running this.
```
./setup.sh
```

**3. Install ```fcc_study```**

To install the fcc_study library, run
```  
pip install -e .  
```  

That's it!

To run the network, all you really need to do is make edits to ```main.py```.
First, at the start of ```main.py``` you will see:
```
######################## Define Hyperparams and Model #########################
base_run_dir = "runs/fcc_scan"
run_loc = getRunLoc(base_run_dir)
```

Give ```base_run_dir``` which is where the training and testing data will be saved. Note that if you don't change ```base_run_dir``` that is fine, a subdirectory is made under it in the for run1, run2, run3... and so on.

Next, you will want to put all of the signal samples into the sample 
dictionary, along with all the background samples. This is of the form: 

```
samples = {
    "backgrounds" : {
    "p8_ee_ZZ_ecm240": {
        "files" : ['p8_ee_ZZ_ecm240.root'], 
        "xs": 1.35899
    },
    "wzp6_ee_eeH_ecm240": {
        "files" : ["wzp6_ee_eeH_ecm240.root"],
        "xs": 0.0071611
    }
    },
    "signal" : {
        "BP1" : {
            "files" : ["e240_bp1_h2h2ll.root", "e240_bp1_h2h2llvv.root"],
            "masses" : [80, 150],
            "xs": 0.0069
        },
        "BP2" : {
            "files" : ["e240_bp2_h2h2ll.root", "e240_bp2_h2h2llvv.root"],
            "masses" : [80, 160],
            "xs": 0.005895
        },
    },
    "Luminosity" : 500,
    "test_size" : 0.25 # e.g. 0.2 means 20% of data used for test set
    }
```

So here you need to fill in multiple things:
1. Put all of the backgrounds in under ```backgrounds```. For 
each background you need to put the list of files (if it's just 1 file then 
still put it in a list like shown), and the cross-section, ```xs```. Note that 
all the background names in the dictionary need to be unique 
(here they are p8_ee_ZZ_ecm240 and wzp6_ee_eeH_ecm240)
2. Do the same for the signal samples, but also include mH and mA for each one 
in masses: ```"masses" : [mH, mA]```. Each signal point needs to have a unique 
name (here BP1 and BP2)
3. Put the correct luminosity, here I just put 500 but you can change that.
4. Then you can pick how large you want the test dataset to be, 0.25 means 
25% of the full dataset is used for the test dataset.

EXTRA NOTE: I think if you are running this on lxplus and on the CERN batch 
system, then when you put in the xrootd file location instead so that 
you can read the file from the batch system.

Note that this uses the cross-section to get the sample weight with the following 
formula: 
```
weight = xs * lumi / n_samples
```

Next you want to put all in the hyper parameters for training. Really the only thing you will have to change here now is the number of epochs, the rest 
you can leave as is. 

Once it has ran, the output is under ```runs```.

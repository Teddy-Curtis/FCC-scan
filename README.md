# FCC Study

Repo for FCC study analysis. This centres around a parametric neural network 
that is used for signal extraction. 

**1. Clone this repository**  

```
git clone git@github.com:Teddy-Curtis/FCC-scan.git
```

**2. Install dependencies**  

All the dependencies are stored in ```environment.yml``` and can be installed 
using conda, mamba or micromamba. 

I used conda, but if you want to install with conda then just replace 
```conda``` with ```micromamba```. Note: You will also have to do the same 
swap in the ```setup.sh``` file as well.

To install with conda:
```
conda env create -f environment.yml
```

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
Here, you will first want to put all of the signal samples into the sample 
dictionary, along with all the background samples. This 
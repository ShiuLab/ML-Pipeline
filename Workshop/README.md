# Instructions for getting the ML-Pipeline Running on Jupyter Notebook

## Install Anaconda and ML-Pipeline dependence packages
1. Download the newest version of [Anaconda](https://www.anaconda.com/distribution/) (python 3).
2. Install Anaconda using the instructions on their webpage.
3. Install conda extension for jupyter (this gives you the conda tab in the Jupyter Notebook page):
```
conda install nb_conda
```
4. Open terminal and make a conda environment with all of the packages you will need for the ML pipeline using the following commands:
```
conda create -n ML_Pipeline numpy pandas scikit-learn scipy matplotlib biopython nb_conda
conda activate ML_Pipeline
```


## Install Git and clone ML-Pipeline
1. Create a [GitHub Account](https://github.com/join)
2. Download [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
3. [Configure](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup) Git in your terminal
4. Clone the [ML-Pipeline](https://github.com/ShiuLab/ML-Pipeline):
```
git clone git@github.com:ShiuLab/ML-Pipeline.git
```

## Navigate to ML-Pipeline tutorial 
1. Fire up Jupyter (pre-installed in anaconda) from the command line by typing:
```
jupyter notebook
```
2. In the window that opens, navigate to GitHub/ML_Pipeline/Tutorial and open ML_Pipeline_tutorial.ipynb

# Instructions for getting the ML-Pipeline Running on Jupyter Notebook

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
5. Fire up Jupyter (pre-installed in anaconda) from the command line by typing:
```
jupyter notebook
```
6. In the window that opens, navigate to GitHub/ML_Pipeline and open a new notebook using 

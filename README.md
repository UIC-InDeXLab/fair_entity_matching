# Fair Entity Matching

### A fairness suite for auditing Entity Matching approaches
Companion repository for the paper "Through the Fairness Lens: Experimental Analysis and Evaluation of Entity Matching". 


## Publication(s) to cite:
[1] Nima Shahbazi, Nikola Danevski, Fatemeh Nargesian, Abolfazl Asudeh, and Divesh Srivastava. "Through the Fairness Lens: Experimental Analysis and Evaluation of Entity Matching." Proceedings of the VLDB Endowment 16, no. 11 (2023): 3279-3292.

[VLDB Publication] <a href="https://dl.acm.org/doi/abs/10.14778/3611479.3611525">https://dl.acm.org/doi/abs/10.14778/3611479.3611525</a> <br>
[Technical Report](techrep.pdf) <br>
[VLDB Slides](FEM-Slides-VLDB23.pdf)



## Installation
- Clone the repo
- Create a virtual environment using e.g., venv or Conda
- Install any missing packages using e.g., pip or Conda
  - main packages are fairly standard (e.g., Pandas, NumPy, SciPy, Scikit-learn, Matplotlib, [Py-EntityMatching](http://anhaidgroup.github.io/py_entitymatching/v0.3.3/index.html))

## Usage
### Familiarize yourself with an example:
fairEM/run_example.py can be used to use our framework to look into the fair behavior of the models on NoFlyCompas dataset.

### Data for reproducing the results:
- Train/Test/Valid/TableA/TableB data for all the datasets in the accepted format by each: [Link](https://drive.google.com/file/d/1ao-IyMffkUsTb5G8I2im9IQraz0mqT6v/view?usp=sharing) 
  - Please note that you do not need these data to reproduce the results. Thes data are only used if interested user wants to (re-)train the used (or any other entitymatching) models in this study.
- Model Predictions: [Link]([https://drive.google.com/file/d/1vJztJVfEh3Rf5QpPBmmyTB55FIY9Z-Ci/view?usp=sharing](https://drive.google.com/file/d/1qG9NC_6HGRbmK3-gEYK-6XijJNfHPgMy/view?usp=sharing)) 
  - Please note that you need this data in order to recreate the results of our study. It includes the predictions of the 13 matchers for 8 datasets. The test sets are also included in the provided link. Test sets are placed in the Deepmatcher folder for each dataset.

### Reproducing the results:
- By putting the provided predictions and test data in the specified locations in fairEM/experiments.py file, the results (plots) of the study
can be generated. 
- fairEM/threshold_experiments.py can be used to regenerate the heatmaps regarding the effect of matching threshold on the fairness and accuracy of the models. 
- fairEM/case_study_analysis.py can be used to look into the model's behavior on specific cases such as TPs, FPs, FNs and TNs.

### Non-neural matchers:
- Examples regarding rule-specifications for rule-based matcher and the settings used for non-neural matchers are brought in utils/entitymatching examples directory.

### Synthetic data generator:
- In synthetic data generator/FacultyMatch and synthetic data generator/NoFlyCompas paths, the scripts that can be used to generate synthetic socail data for entity matching are provided. Users can employ these scripts to create such datasets with a variety of settings such as limiting the rate on non-matches in the output (i.e. manual blocking), change the number and type of perturbations and etc.

## Notice
This project is still under development, so please beware of potential bugs, issues etc. Use at your own responsibility in practice.

## Contact
Feel free to contact the authors or leave an issue in case of any complications. We will try to respond as soon as possible.

## License

This project is licensed under the MIT License &mdash; see the [LICENSE.md](LICENSE.md) file for details.

<p align="center"><img width="20%" src="https://www.cs.uic.edu/~indexlab/imgs/InDeXLab2.gif"></p>


# Fair Entity Matching
### A fairness suit for auditing Entity Matching approaches
Companion repository for the paper "Through the Fairness Lens: Experimental Analysis and Evaluation of Entity Matching".

### Data for reproducing the results:

Train/Test/Valid/TableA/TableB data for all the datasets in the accepted format by each: [Link](https://drive.google.com/file/d/1ao-IyMffkUsTb5G8I2im9IQraz0mqT6v/view?usp=sharing) (Please note that you do not need these data to reproduce the results. Thes data are only used if interested user wants to (re-)train the used (or any other entitymatching) models in this study.) <br>
Model Predictions: [Link](https://drive.google.com/file/d/1vJztJVfEh3Rf5QpPBmmyTB55FIY9Z-Ci/view?usp=sharing) (Please note that you need this data in order to recreate the results of our study. It includes the predictions of the 13 matchers for 8 datasets. The test sets are also included in the provided link. Test sets can be are placed in the Deepmatcher folder for each dataset.) <br>

### Familiarize yourself with an example:
fairEM/run_example.py can be used to use our framework to look into the fair behavior of the models on NoFlyCompas dataset.

### Reproducing the results:
By putting the provided predictions and test data in the specified locations in fairEM/experiments.py file, the results (plots) of the study
can be generated. fairEM/threshold_experiments.py can be used to regenerate the heatmaps regarding the effect of matching threshold on the fairness and accuracy of the models. fairEM/case_study_analysis.py can be used to look into the model's behavior on specific cases such as TPs, FPs, FNs and TNs.

### Non-neural matchers:
Examples regarding rule-specifications for rule-based matcher and the settings used for non-neural matchers are brought in utils/entitymatching examples directory.

### Synthetic data generator:
In synthetic data generator/FacultyMatch and synthetic data generator/NoFlyCompas paths, the scripts that can be used to generate synthetic socail data for entity matching are provided. Users can employ these scripts to create such datasets with a variety of settings such as limiting the rate on non-matches in the output (i.e. manual blocking), change the number and type of perturbations and etc.


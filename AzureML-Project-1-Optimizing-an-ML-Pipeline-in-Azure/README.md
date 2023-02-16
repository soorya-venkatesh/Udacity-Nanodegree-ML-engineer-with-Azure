# AzureML-Project 1-Udacity Machine Learning Engineer Nanodegreee Program

## Problem Statement:
The data is a bank marketing dataset.The Problem statement is to predict whether a customer will make a deposit or not (binary classification problem) given a set of attributes. 


## Objective:
The following are the objectives for this project
- To train a logistic regression model in AzureML and tune its hyperparameters using Hyperdrive.
- To set AutoML in AzureML and come up with a best model for the problem.
- To compare and contrast the above two approaches. 

## The dataset used:

The data contains 32950 rows of people containing carious features .Some of the features include age, martial status,job ,has taken a loan or not and correspondiong date information like month,week etc. It also contains the target variable "Y" which contains 2 values - Yes/No. This target variable must be related to the response of these customers to some marketing campaign. **The overall objective** would be to predict the target variable 'Y' given the set of features. 

## Overall Solutioning:
- The train.py contains a pipeline which gets dataset using TabularDatasetFactory class, pre processes the data , split into train and test sets and accepts hyperparameters "C" and "max_iter" as arguments which will be used in the hyperdrive config setup. 
- In udacity-project.ipynb , a workspace is and an experiment are created. Then a compute cluster is created as a part of initial set up.
    - For Hyper drive ; the parameter estimation policy, the Sklearn estimator , the hyperdrive config file is set up to train the model and find best hyperparameters, report the accuracy on test data and save the best model.
    - For AutoML ; the data is imported , AutoML config is created which includes a 5 fold cross validation and run to find best model, find the accuracy on test data and save the best model.

## Hyperdrive Experiment Details:
### Type of classification algorithm used:
The algorithm used for this classification problem was logistic regression. The hyperparameters used for this were C (which is inverse of the regularization strength) and max_iter(number of maximum iterations which can be used for convergence )

### Parameter Sampling Stratergy:
Random Search was used to find the best value of "C" and "max_iter". The search space for "C" is [0.0001,1000] and search space for "max_iter" was {75,125}.
#### Benifits of Parameter Sampling:
In any machine learning problem in order to find the optimal value(s) of weights , a search stratergy is used. Some of them are Random search (picking hyperparameter combinations at random and choosing the one with least loss or best metric ) , grid search (iteratively going through all possibilities of hyperparameter combinations and choosing the one with the least loss or best metric) , Bayesian search (using Bayesian stratergies to search for the best hyperparameter combination) etc.  

### Early Stopping Policy:
A bandit policy of slack factor=0.17 and evaluation_interval=2 is being used. A slack factor=0.17 will terminate the training when if the metric at the current iteration is 17% less than the best performing metric. evaluation_interval=2 means that this evaluation will happen once every 2 iterations.  

### Hyperparameter Tuning and Result:
The best value of **"C" was 693.1** and best value of **"max_iter" was 125**. The **best accuracy gotten was 0.918**

![Alt text](static/tuned_paramters.png?raw=true "Title")


## AutoML Experiment Details:
### AutoML config:

![Alt text](static/automl_config.PNG?raw=true "Title")
- The AutoML is run for 30 minutes with 30 iterations meaning 30 different combinations of models are tried.
-  A 5 fold cross validation was used to test the data instead of a ordinary train-test split.
-  The task given was 'classification' since this was a binary classification problem
-  The primary metric of intrest was given to be 'accuracy'
-  The path for the dataset ('training_data') and the and the target column in the data ('label_column_name') was also mentioned.

For full list of available parameters in AutoML config , the official documentation link can be found [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train).

### Run Details:
The below screenshots reprsents the models used by AutoML along with its training time and metric. 
![Alt text](static/automl_result_1.png?raw=true "Title1")

![Alt text](static/automl_result_2.png?raw=true "Title2")
### Model parameters generated by AutoML:
The AutoML ran through various classification algorithms which included Random forest Classifier,Support Vector Classifier, XGBoost Classifier , voting ensemble classifer wach having its own parameters to tune. 
For Instance consider random forest and its main parameters gotten from this experiment
- criterion: talk about the the critera for splitting the nodes in the tree (in this experiment criteria=gini)
- n_estimators: Describes the maximum number of decision trees which can be used (in this experiment n_estimators=200)
- max_depth: Describes the max depth of each decision tree (this experiment max depth:4)

### Future Work:
- For the hyperdrive config, a Bayesian search can be used to check out the results 
- For Auto ML, increasing the number of iterations and total training time could lead to better results. So that can be a intresting experiment to try out.
- Trying to increase increase the number of folds in the cross validation for the Auto ML 

### Result:
A **maximum accuracy of 0.918** was got using a **voting ensemble classifier**. Other metrics are also shown below.
![Alt text](static/automl_detailed_metrics.png?raw=true "Title2")

### Best Model description:
A voting ensemble classifier takes multiple models into consideration. It happens in two stages. Stage 1 is gets the predictions from all the models taken into consideration. Stage 2 takes all the predictions at the end of stage 1 and does a voting. The Majority voted class is the final predicted class.

## Comparision of results and inference:
- Both the logistic regression and AutoML yielded the same accuracy of 0.918 on test data
- AutoML was trained only for 30 minutes. Training for a longer time would have yielded better results.
- Training the logistic regression was took less time than AutoML.

## References:
- https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_factory.tabulardatasetfactory?view=azure-ml-py#azureml_data_dataset_factory_TabularDatasetFactory_from_delimited_files
- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters
- https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py
- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train


## Deletion of compute cluster:

![Alt text](static/delete_compute.png?raw=true "Title2")

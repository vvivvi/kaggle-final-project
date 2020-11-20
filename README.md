# kaggle-final-project
Final project of the course" How to Win a Data Science Competition: Learn from Top Kagglers" by National Research University Higher School of Economics

## Overview of the approach

This solution does not aim for the best possible prediction accuracy. Instead, a straightforward solution is described that involves most of the techniques mentioned in the final project requirements. 

### Task

### Approach
In the  implemented approach
1. The day-level transactional data is aggregated into monthly summaries for each (shop,item) combination. Each monthly summary is considered as a row in a data matrix.  
2. Each row is augmented with additional features describing the particular item and shop. In particular, the items are grouped into categories derived from the item name text with help of text processing techniques, along with the provided item categories. 
3. Each row is augmented with temporally lagged versions of the target variable, i.e. the monthly sales count. In addition to sales history specific to each (item,shop) combination separately, we include also sales history aggregated over the item categories that were added in step 2, both on the level of individual items and an the shop level.
4. Sequential feature selection is performed to select a few subsets of the generated features. The features are selected to maximise the performance of CatBoost regression
models in a validation experiment.
5. The same validation setup is used for searching  optimal hyperparameters for both CatBoost and Random Forest regression models.
6. CatBoost and Random Forest regression models are trained for predicting the target month (December 2015) sales, based on each of the selected feature subsets.
7. The predictions of the models are ensembled using a simple averaging scheme.
8. Data leaking through the leaderboard is used for scaling 
the ensembled predictions optimally.



## Organisation of the work

List of notebooks
* EDA
* data leak analysis
* Feature generation
* Feature selection and hyperparameter tuning
* Prediction model training and prediction generation 
* Ensebling of predictions
* Prediction scaling

## Requirements checklist

## Result summary

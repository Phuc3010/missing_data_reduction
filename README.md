# Misisng Data Imputation
## Step 1: Installition
Install the necessary packages using the following command:
```shell
pip install -r requirements.text
```
## Step 2: Running
```shell
python main.py --classifier {classifier_name} --reduce_method {reduce_method} --missing_rate {missing_rate} --dataset {dataset} --impute_method {impute_method} --non_missing {non_missing}
```
Where
- `classifier_name` The name of the classifier, can be `logistic, random_forest, neural_nets`.
- `reduce_method`: Dimension reduction methods, can be `pca, svd, kernelpca`.
- `missing_rate`: The missing rate for simulation. This should be set either 0.2, 0.4 or 0.6.
- `dataset`: dataset name, can be `MNIST, CIFAR10, churn_risk, credit_card, IMDB`.
- 'impute_method`: Select imputation algorithms, the available algorithms: `softimpute, mice, gain, knn`.
- `non_missing`: Non misisng columns for dimension reduction, must be smaller than the feature dimension of the inputs.

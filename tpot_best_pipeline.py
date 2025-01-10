import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.impute import SimpleImputer

# Load the dataset
tpot_data = pd.read_csv(
    '/content/gdrive/MyDrive/Colab Notebooks/Data/CMAPSSData/train_FD001.txt',
    sep=' ',
    header=None,
    skipinitialspace=True
)

# Drop extra columns
tpot_data.dropna(axis=1, how='all', inplace=True)

# Rename columns for clarity
columns = [f"feature_{i}" for i in range(tpot_data.shape[1] - 1)] + ["RUL"]
tpot_data.columns = columns

# Binarize the target column
failure_threshold = 30  # Define RUL threshold for failure
tpot_data['target'] = tpot_data['RUL'].apply(lambda x: 1 if x <= failure_threshold else 0)

# Drop the original RUL column
tpot_data.drop(columns=['RUL'], inplace=True)

# Impute missing values with the median
imputer = SimpleImputer(strategy="median")
tpot_data[:] = imputer.fit_transform(tpot_data)

# Split into features and target
features = tpot_data.drop('target', axis=1)
target = tpot_data['target']

# Train-test split
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, target, random_state=42, stratify=target)

# Define the pipeline
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=72),
    StackingEstimator(estimator=BernoulliNB(alpha=100.0, fit_prior=False)),
    RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.5, n_estimators=100), step=0.65),
    RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.85, min_samples_leaf=13, min_samples_split=3, n_estimators=100)
)

# Fix random state for reproducibility
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

# Train and evaluate the pipeline
exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

# Print results
print("Pipeline executed successfully!")

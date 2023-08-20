#!/usr/bin/env python
# coding: utf-8

# In[75]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import ttest_rel
import scikitplot as skplt
from collections import Counter
import joblib

from sklearn.model_selection import (
    learning_curve, cross_val_score, train_test_split,
    RandomizedSearchCV, GridSearchCV, KFold, StratifiedKFold
)
from sklearn import naive_bayes, svm, neighbors
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    auc, roc_auc_score, precision_score, accuracy_score,
    recall_score, f1_score, plot_confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier,
    RandomForestClassifier, StackingClassifier
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.base import clone
import re
import base64


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')

# Custom colors
class clr:
    S = '\033[1m' + '\033[36m'
    E = '\033[0m'


# In[76]:


# FUNCTIONS

# Distribution of spam/ham column where 1 is spam and 0 is not spam
def print_distribution(data, title):
    counts = data['spam/ham'].value_counts()
    percentages = counts / len(data) * 100
    print(title)
    for index in counts.index:
        print(f"{index}: {counts[index]} ({percentages[index]:.2f}%)")
    print("\n")
    
# Function for counting number of correlations a feature label has with other feature labels on a threshold value.
def count_high_correlations(df, threshold=0.3):
    corr_matrix = df.corr()
    column_counts = {}

    for col in corr_matrix.columns:
        # Count how many other features a feature is highly correlated with
        high_corr_count = sum(corr_matrix[col].apply(lambda x: abs(x) > threshold)) - 1  # Subtract 1 to exclude itself
        column_counts[col] = high_corr_count

    return column_counts
            
# Function to train a model and obtain its feature importances
def get_importances(model, X, y):
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return importances, indices

# Convert feature lists to DataFrames
def create_feature_df(importances, indices, num_features, label):
    return pd.DataFrame({
        'feature_label': [df.columns[i] for i in indices[:num_features]],
        label: [importances[i] for i in indices[:num_features]]
    })

    return column_counts

# Feature importance bar plots
def display_feature_importance(importances, indices, title):
    plt.figure(figsize=(15, 5))
    plt.title(title)
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [df.columns[i] for i in indices], rotation='vertical')
    plt.xlim([-1, len(indices)])
    plt.show()


# ROC_AUC plots for feature importances
def feature_plots(X_train, y_train, X_val, y_val, features_df, feature_col='feature_label'):
    # Use the provided dataframe to decide on top features.
    features_sorted = features_df.sort_values(by=feature_col, ascending=False)

    # Mapping feature names to their respective indices
    feature_indices = {name: index for index, name in enumerate(features_df['feature_label'])}
    
    roc_aucs = []
    roc_curves = []
    
    for n in range(1, len(features_sorted) + 1):
        top_features = features_sorted['feature_label'].head(n)
        top_feature_indices = [feature_indices[feature] for feature in top_features]
        
        X_train_sub = X_train.iloc[:, top_feature_indices]
        X_val_sub = X_val.iloc[:, top_feature_indices]

        clf = RandomForestClassifier()
        clf.fit(X_train_sub, y_train)
        y_pred_proba_val = clf.predict_proba(X_val_sub)[:, 1]
        
        # Calculate and store ROC AUC
        roc_aucs.append(roc_auc_score(y_val, y_pred_proba_val))
        
        # Calculate and store ROC curve
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba_val)
        roc_curves.append((fpr, tpr))

    # Create subplots    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    
    # Performance vs number of features
    ax1.plot(roc_aucs)
    ax1.set_xlabel("Number of Features")
    ax1.set_ylabel("ROC AUC on Validation Set")
    ax1.set_title("Performance vs Number of Features")

    # ROC curves for various numbers of features
    for i, (fpr, tpr) in enumerate(roc_curves[::5]):  
        ax2.plot(fpr, tpr, label=f"Top {5*i + 1} features")
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel("False Positive Rate (FPR)")
    ax2.set_ylabel("True Positive Rate (TPR)")
    ax2.set_title("ROC Curves for Different Number of Features")
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()

#  For Datasets
def cross_val_dataset(X, y, test_size=0.2, random_seed=None):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_seed)
    return X_train, X_val, y_train, y_val

# ROC Curves function
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    
    for name, model in models:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

#  Randomized Hyperparameter Tuning Function
def randomized_hyperparameter_tuning(model, param_distributions, X_train, y_train, n_iter=10, cv=5):
    random_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=n_iter, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1, random_state=42)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_params_ 
    return best_model


#  Model Evaluation Function
def evaluate_complex(model, X_val, y_val):
    y_pred = model.predict(X_val)
    # Accuracy
    acc = accuracy_score(y_val, y_pred)
    # Classification Report
    report = classification_report(y_val, y_pred)
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    print(f"Accuracy: {acc}\n")
    print(f"Classification Report:\n{report}\n")
    print(f"Confusion Matrix:\n{conf_matrix}\n")
    return acc, report, conf_matrix

#  Model Evaluation Function
def evaluate_simple(model, X_val, y_val):
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    return acc

# Fit and predict models with cross validationdf
def fit_and_predict(model, X_train, y_train, X_test):
    # Fit the model on the training data
    model.fit(X_train, y_train)
    # Make predictions on the training data
    y_pred_train = model.predict(X_train)
    # Make predictions on the test data
    y_pred_test = model.predict(X_test)
    # Number of mislabeled points in the training set
    print("Number of mislabeled points out of a total of %d points : %d" % (X_train.shape[0], (y_train != y_pred_train).sum()))
    # Empirical error over 10 folds
    print("Empirical error over 10 folds: {:.2%}".format((y_train != y_pred_train).sum()/X_train.shape[0]))
    # Calculate cross-validation score over 10 folds
    scores = cross_val_score(model, X_train, y_train, cv=25, n_jobs=8, scoring='roc_auc')
    print("Cross-validation score over 10 folds : {:.3%}".format(np.mean(scores)))
        
    return y_pred_train, y_pred_test

# Resample
def bootstrap_confidence_interval(model, X_val, y_val, n_iterations=1000, ci=0.95):
    accuracies = []
    for _ in range(n_iterations):
        # Sample with replacement from X_val and y_val
        X_resample, y_resample = resample(X_val, y_val)
        acc = evaluate_simple(model, X_resample, y_resample)
        accuracies.append(acc)

    # Calculate lower and upper percentiles for CI
    lower = ((1.0 - ci) / 2.0) * 100
    upper = (ci + ((1.0 - ci) / 2.0)) * 100
    lower_bound = np.percentile(accuracies, lower)
    upper_bound = np.percentile(accuracies, upper)
    
    return lower_bound, upper_bound

# Effect size function
def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

# Perform Paired T-tests between two models
def perform_paired_t_test(model_1, model_2, X_val, y_val):
    y_pred1 = model_1.predict(X_val)
    y_pred2 = model_2.predict(X_val)
    
    # Assuming accuracy is the metric of interest
    errors_model_1 = [0 if true == pred else 1 for true, pred in zip(y_val, y_pred1)]
    errors_model_2 = [0 if true == pred else 1 for true, pred in zip(y_val, y_pred2)]
    
    t_stat, p_value = ttest_rel(errors_model_1, errors_model_2)
    return t_stat, p_value

# ROC curve
def plot_roc_curve(model, X_test, y_test, label):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
# Learning Curve   
def plot_learning_curve(model, X, y, title):     # the classifier, features, target variable, title for the plot

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring="accuracy", n_jobs=-1, 
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot the learning curve
    plt.plot(train_sizes, train_mean, label="Training score", color="blue")
    plt.plot(train_sizes, test_mean, label="Cross-validation score", color="red")
    
    # Shade the region of +/- std deviation
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="red", alpha=0.1)
    
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    
# Calibration Curve
def plot_calibration_curve(model, X, y, title):

    # Predict probabilities for the positive class
    prob_pos = model.predict_proba(X)[:, 1]
    
    # Get calibration curve values
    prob_true, prob_pred = calibration_curve(y, prob_pos, n_bins=10, strategy="uniform")
    
    # Plot the calibration curve
    plt.plot(prob_pred, prob_true, marker="o", label="Model", color="blue")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    
    plt.title(title)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend(loc="best")
    
# Helper function to compute rates
def compute_rates(cm):
    TP, FN, FP, TN = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]
    
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (TP + FN)
    
    return TPR, FPR, TNR, FNR

# Plot confusion matrix rates
def plot_cm_with_rates(ax, cm, TPR, FPR, TNR, FNR):
    """Annotate the ax with rates."""
    ax.text(0, 0.25, f'TNR: {TNR:.2f}', ha='center', va='center', color='red', fontsize=11)
    ax.text(0, 1.25, f'FPR: {FPR:.2f}', ha='center', va='center', color='red', fontsize=11)
    ax.text(1, 0.25, f'FNR: {FNR:.2f}', ha='center', va='center', color='red', fontsize=11)
    ax.text(1, 1.25, f'TPR: {TPR:.2f}', ha='center', va='center', color='red', fontsize=11)


# In[77]:


# Importing and Reading Datasets
# Replace this path with the actual path to your data_folder
data_folder = r'C:\Users\ljfit\Desktop\UCDPA-L.J._Fitzgerald\spambase'

# Read feature names from the names file
names_file_path = os.path.join(data_folder, 'spambase.names')
with open(names_file_path, 'r') as names_file:
    lines = names_file.readlines()
    feature_names = [line.strip().split(':')[0] for line in lines[33:]]

# Read data from the .data file and join with .names files
data_file_path = os.path.join(data_folder, 'spambase.data')
nmpy = pd.read_csv(data_file_path, header=None, names=feature_names)

# Display the first few rows of the combined DataFrame
print(nmpy.head())

# This will convert the column names to a list
column_names = nmpy.columns.tolist()
print(column_names)

print("\n")

# Check if column headers and data line up
if len(nmpy.columns) == len(feature_names):
    print("Column headers and data line up.")
else:
    print("Column headers and data do not line up.")


# In[78]:


# Descriptive statistics
print(nmpy.shape)
print(nmpy.describe())
print(nmpy.info())


# In[79]:


# Distribution of spam/ham column
print(nmpy['spam/ham'].value_counts())
# Get value counts as percentages
percentages = nmpy['spam/ham'].value_counts(normalize=True) * 100

# Print the percentages
print(percentages)


# In[80]:


# Create a bar plot using the calculated percentages
fig, ax = plt.subplots(figsize=(10, 10))

# Adjust subplot parameters
fig.subplots_adjust(left=0.2, right=1.0, top=0.95, bottom=0.2)

# Plot the percentages with seaborn
sns.barplot(x=percentages.index, y=percentages, ax=ax, palette="viridis")

# Modify x-tick labels
ax.set_xticklabels(['ham: 0', 'spam: 1'])

# Add percentages inside the bars
for i, v in enumerate(percentages):
    ax.text(i, v/2, "{:.1f}%".format(v), ha='center', size=12, color='white', fontweight='bold')

# Plot
ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage', fontsize=12, fontweight='bold')
plt.title('Distribution of ham and spam emails', fontsize=16, fontweight='bold')

# Show the plot
plt.show()
# No class imbalance


# In[81]:


# Convert numpy array to a Dataframe
df = pd.DataFrame(nmpy)

# # Check for missing values
# print(df.isnull().sum())

# Check for duplicate columns
print(df.T.duplicated().sum())

# Check for duplicate rows
print(df.duplicated().sum())

# Rows that contain only zeros
print((df == 0).all(axis=1).sum())

# Filter only duplicated rows
duplicated_rows = df[df.duplicated()]

print(df.shape)


# In[82]:


# Extract duplicated rows
duplicated_rows = df[df.duplicated()]

# Filtered rows based on different conditions
rows_3 = duplicated_rows[duplicated_rows.max(axis=1) <= 3]
rows_100 = duplicated_rows[duplicated_rows.max(axis=1) <= 100]
rows_15841 = duplicated_rows[duplicated_rows.max(axis=1) <= 15841]

# Print distributions
print_distribution(rows_3, "Distribution for duplicate rows with max value <= 3:")
print_distribution(rows_100, "Distribution for duplicate rows with max value <= 100:")
print_distribution(rows_15841, "Distribution for duplicate rows with max value of 15841:")
print_distribution(df, "Overall distribution:")


# In[83]:


# Create a heatmap of the correlation matrix
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(df.corr(), dtype=bool))

plt.figure(figsize=(20, 20))
plt.subplots_adjust(left=0.2, right=1.0, top=0.95, bottom=0.2)

# Apply the mask and cmap for the heatmap
sns.heatmap(df.corr(), mask=mask, cmap='coolwarm')

plt.show()


# In[84]:


# PCA on the original Dataset
print("df type:", type(df))
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
pca = PCA()
df_pca = pca.fit_transform(df_scaled)

# Find the number of components that retain 95% variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance >= 0.95) + 1  
print(f"Number of components that retain 95% variance: {num_components}")

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, len(pca.explained_variance_ratio_)+1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance')
plt.axvline(x=num_components, color='g', linestyle='--', label=f'{num_components} components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title("PCA for df")
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[85]:


# Extract features and target from the dataframe
X = df.drop('spam/ham', axis=1)    # Features
y = df['spam/ham'].values          # Target labels

# Creating Correlation count dataframe
high_corr_counts = count_high_correlations(X, 0.3)

# Convert dictionary to DataFrame
count_df = pd.DataFrame(list(high_corr_counts.items()), columns=['feature_label', 'High Correlation Count'])
count_df_features = count_df.sort_values(by='High Correlation Count', ascending=False)
reduced_count_df = count_df_features.iloc[:, :30]
print(reduced_count_df)


# In[86]:


# Define the number of features to select for each method
num_features_rf = 57
num_features_gb = 57
num_features_rg_gb = 57

# Train RandomForest and obtain its feature importances
rf_importances, rf_indices = get_importances(RandomForestClassifier(n_estimators=100), X, y)

# Train GradientBoosting and obtain its feature importances
gb_importances, gb_indices = get_importances(GradientBoostingClassifier(n_estimators=100), X, y)

# Sort 
rf_features_df = create_feature_df(rf_importances, rf_indices, num_features_rf, 'RF_Importance')
gb_features_df = create_feature_df(gb_importances, gb_indices, num_features_gb, 'GB_Importance')

# Display the RandomForest feature importances.
display_feature_importance(rf_importances, rf_indices, "Feature Importances (RandomForest)")

# Display the GradientBoosting feature importances.
display_feature_importance(gb_importances, gb_indices, "Feature Importances (GradientBoosting)")

# Store the DataFrames in a dictionary
dataframes = {
    "RF Features": rf_features_df,
    "GB Features": gb_features_df,
    "Count Features": reduced_count_df
}


# In[62]:


# ROC_AUC plotting for feature importances
cv = StratifiedKFold(n_splits=2)

# Iterate over each feature importance dataframe
for name, features_df in dataframes.items():
    print(f"Processing for: {name}")
    
    for train_index, val_index in cv.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Use the current feature importance dataframe in the function
        feature_plots(X_train, y_train, X_val, y_val, features_df, feature_col='feature_label')


# In[67]:


# Assign top-n features to variables for further use
rf_features = dataframes["RF Features"].head(30)
gb_features = dataframes["GB Features"].head(30)
count_df = dataframes["Count Features"].head(30)
print(rf_features)
print(gb_features)
print(count_features)

# Store the DataFrames in a dictionary
dataframes = {
    "RF Features": rf_features,
    "GB Features": gb_features,
    "Count Features": count_df
}


# In[68]:


# Descriptive statistics for the feature variables
for name, data in dataframes.items():
    # Display the shape of the current DataFrame
    print(f"{name} Shape of DataFrame with selected features: {data.shape}")
    
    # Get descriptive statistics
    print(f"\n{name} Descriptive Statistics of the DataFrame with selected features:\n")
    print(data.describe())
    print('\n')
    
    # Print DataFrame shape for reference
    print('-' * 50)  # line separator for clarity


# In[47]:


# Using the RF features importance values for the rest of the study.     ### COMMENT OUT AND RUN FROM HERE ###
selected_features1 = rf_features['feature_label'].tolist()               ### FOR DIFFERENT DATASETS ###
X = X[selected_features1]                                                ### COMMENT OUT ALL TO RUN ORIGINAL DATASET ###

# # Using the GB features importance values for the rest of the study.
# selected_features1 = gb_features['feature_label'].tolist()
# X = X[selected_features]
# print(X)


# In[18]:


# Evaluating base model performance
# Prepare algorithms to evaluate
base_models = {
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(gamma='auto', probability=True))
    ]),
    'GNB': GaussianNB(),
    'BNB': BernoulliNB()
}

# Assuming you already have your cross_val_dataset function and your evaluate_model function defined elsewhere
# Splitting the data
X_train_base, X_val_base, y_train_base, y_val_base = cross_val_dataset(X, y)

acc_base = []

# Training and Evaluating on Feature Data
for name, model in base_models.items():
    model.fit(X_train_base, y_train_base)
    
    print(f"\nEvaluating {name}...\n")
    accuracy, _, _ = evaluate_complex(model, X_val_base, y_val_base) # This line has changed
    acc_base.append(accuracy)
    
# After all models are evaluated, plot ROC curves on the same graph
plot_roc_curves(base_models.items(), X_val_base, y_val_base)

# Shapes of datasets
print("X Train shape:", X_train_base.shape)
print("X Val shape:", X_val_base.shape)
print("y Train shape:", y_train_base.shape)
print("y Val shape:", y_val_base.shape)


# In[21]:


# Confidence Intervals without Hypertuning
for name, model in base_models.items():
    print(f"Computing confidence interval for {name}...")
    lower, upper = bootstrap_confidence_interval(model, X_val_base, y_val_base)
    print(f"95% Confidence Interval for {name} Accuracy: [{lower:.3f}, {upper:.3f}]")


# In[22]:


# Evaluating hypertuned model performance
# Algorithms to evaluate
hyp_models = {
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(gamma='auto', probability=True))  # probability=True for ROC curve plotting
    ]),
    'GNB': GaussianNB(),
    'BNB': BernoulliNB(),
}

# Parameters for hypertuning
param_distributions = {
    'KNN': {'classifier__n_neighbors': [1, 5, 11, 15, 21]},
    'SVM': {'classifier__C': [0.1, 1.0, 2.0]},
    'GNB': {},
    'BNB': {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0]}
}

# Copy of split data
X_train_hyp = X_train_base.copy()
X_val_hyp = X_val_base.copy()
y_train_hyp = y_train_base.copy()
y_val_hyp = y_val_base.copy()

# Lists to append to
optimized_models = []
acc_hyp = []

# Training and Evaluating on Hypertuned Data
for name, model in hyp_models.items():
    # Hyperparameter tuning
    best_params = randomized_hyperparameter_tuning(model, param_distributions[name], X_train_hyp, y_train_hyp)
    
    # Refitting model with best parameters and evaluating
    optimized_model = model.set_params(**best_params)
    optimized_model.fit(X_train_hyp, y_train_hyp)  
    
    print(f"\nEvaluating {name} with optimized parameters...\n")
    # Evaluating the model using evaluate_on_validation_set function
    accuracy, _, _ = evaluate_complex(optimized_model, X_val_hyp, y_val_hyp)
    acc_hyp.append(accuracy)
    
    # Store the refitted model and its name
    if hasattr(optimized_model, "predict_proba"):
        optimized_models.append((name, optimized_model))

# After all models are evaluated, plot ROC curves on the same graph
plot_roc_curves(optimized_models, X_val_hyp, y_val_hyp)
    
# Shapes of datasets
print("X Train shape:", X_train_hyp.shape)
print("X Val shape:", X_val_hyp.shape)
print("y Train shape:", y_train_hyp.shape)
print("y Val shape:", y_val_hyp.shape)


# In[23]:


# Confidence Intervals of Original Dataset with Hypertuning
for name, model in hyp_models.items():
    print(f"Computing confidence interval for {name}...")
    lower, upper = bootstrap_confidence_interval(model, X_val_hyp, y_val_hyp)
    print(f"95% Confidence Interval for {name} Accuracy: [{lower:.3f}, {upper:.3f}]")


# In[30]:


# Hypothesis Testing between Base and Hypertuned Model
base_model_name = "KNN"
base_model = base_models[base_model_name]
hyp_model_name = optimized_models[0][0]
hyp_model = optimized_models[0][1] 

# Set the significance level
alpha = 0.025

print(f"\nObjective: To determine if hypertuned {hyp_model_name} offers superior performance to \nthe base {base_model_name} in terms of accuracy.\n")

print(f"H0: The performance of the hypertuned {hyp_model_name} is equivalent to the base {base_model_name}.\n")
print(f"H1: The performance of the hypertuned {hyp_model_name} is significantly better than the base {base_model_name}.\n")

# Paired t-test: Compare hypertuned model performance vs. base model
t_stat, p_value = perform_paired_t_test(hyp_model, base_model, X_val_hyp, y_val_hyp)

# Effect Size calculation
y_pred_base = base_model.predict(X_val_base)
errors_base_model = [0 if true == pred else 1 for true, pred in zip(y_val_base, y_pred_base)]
y_pred_hyp = hyp_model.predict(X_val_hyp)
errors_hyp_model = [0 if true == pred else 1 for true, pred in zip(y_val_hyp, y_pred_hyp)]
effect_size = cohens_d(errors_base_model, errors_hyp_model)

print("p-value:", p_value)

print(f"\n{base_model_name} (Base) vs {hyp_model_name} (Hypertuned):")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Effect size: {effect_size:.4f}")
if p_value < alpha:
    print(f"\nThe performance difference between the hypertuned and the base {base_model_name} is statistically significant at the {alpha*100}% level.\n")
else:
    print(f"\nThere's no significant performance difference between the hypertuned and the base {base_model_name} at the {alpha*100}% level.\n")

print("Conclusion:")
print(f"In this analysis, base and hypertuned classifiers for {base_model_name} were trained. Performances were evaluated and\nstatistically compared. The results provide insights into whether hypertuning offers additional value\ncompared to the base model settings.\n")


# In[32]:


# Base Models: Confusion Matrices
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
fig.suptitle('Confusion Matrices for Base Models')
axes = axes.ravel()

for idx, (model_key, model) in enumerate(base_models.items()):
    cm = confusion_matrix(y_val_base, model.predict(X_val_base))
    TPR, FPR, TNR, FNR = compute_rates(cm)
    
    # Plot
    plot_confusion_matrix(model, X_val_base, y_val_base, display_labels=[f'TNR:{TNR:.2f}', f'TPR:{TPR:.2f}'], cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'Base {model_key}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

# Optimized models: Confusion Matrices
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
fig.suptitle('Confusion Matrices for Hyper-Tuned Models')
axes = axes.ravel()

for idx, (model_key, model) in enumerate(optimized_models):
    cm = confusion_matrix(y_val_hyp, model.predict(X_val_hyp))
    TPR, FPR, TNR, FNR = compute_rates(cm)
    
    # Plot
    plot_confusion_matrix(model, X_val_hyp, y_val_hyp, display_labels=[f'TNR:{TNR:.2f}', f'TPR:{TPR:.2f}'], cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'Hyper-Tuned {model_key}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

print(f"{model_key} base predictions:", model.predict(X_val_base)[:10])
print(f"{model_key} optimized predictions:", model.predict(X_val_hyp)[:10])


# In[73]:


rates = []
# Compute and store rates for base models
for model_key, model in base_models.items():
    cm = confusion_matrix(y_val_base, model.predict(X_val_base))
    TPR, FPR, TNR, FNR = compute_rates(cm)
    rates.append({"Model": f"Base {model_key}", "TPR": TPR, "FPR": FPR, "TNR": TNR, "FNR": FNR})

# Compute and store rates for optimized models
for idx, (model_key, model) in enumerate(optimized_models):
    cm = confusion_matrix(y_val_hyp, model.predict(X_val_hyp))
    TPR, FPR, TNR, FNR = compute_rates(cm)
    rates.append({"Model": f"Hyper-Tuned {model_key}", "TPR": TPR, "FPR": FPR, "TNR": TNR, "FNR": FNR})

# Convert rates to DataFrame
rates_df = pd.DataFrame(rates)

# Display rates table
print(rates_df)


# In[33]:


# Ensemble Models Initialization
ensemble_models = {
    'AB': AdaBoostClassifier(),
    'RF': RandomForestClassifier(n_estimators=10),
    'GB': GradientBoostingClassifier()
}

# Split data
X_train, X_val, y_train, y_val = cross_val_dataset(X, y)

# Train and evaluate models on validation data
names_ensemble = []
results_ensemble = []
for name, model in ensemble_models.items():
    model.fit(X_train, y_train)
    acc, _, _ = evaluate_complex(model, X_val, y_val)
    names_ensemble.append(name)
    results_ensemble.append(acc)

# Visualization of Model Performance
fig, ax = plt.subplots(figsize=(20,7))
palette = sns.color_palette("Set2", len(ensemble_models))
sns.barplot(x=names_ensemble, y=results_ensemble, palette=palette, ax=ax)
ax.set_ylabel("Performance Metric (e.g., Accuracy)")
ax.set_xlabel("Model Name")
plt.title('Ensemble Algorithm Comparison')
plt.grid(axis="y")
plt.show()

# Display shape of datasets
print("X Train shape:", X_train.shape)
print("X Test shape:", X_val.shape)
print("y Train shape:", y_train.shape)
print("y Test shape:", y_val.shape)


# In[34]:


# Display model performance on validation set
print("\nPerformance on Validation Set:")
for name, score in zip(names_ensemble, results_ensemble):
    print(f"{name}: {score:.4f} Accuracy")

# Cross-Validation Scores
print("\nCross-Validation Scores:")
for name, model in ensemble_models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")


# In[35]:


# Bootstrapped Confidence Intervals
print("\nBootstrap Confidence Intervals:")
for name, model in ensemble_models.items():
    lower, upper = bootstrap_confidence_interval(model, X_val, y_val)
    print(f"{name}: {100*lower:.2f}% - {100*upper:.2f}%")


# In[42]:


# Plotting the learning curves for  
plt.figure(figsize=(15, 20))

for idx, (name, model) in enumerate(ensemble_models.items(), 1):
    plt.subplot(len(ensemble_models), 1, idx)
    plot_learning_curve(model, X, y, title=f"Learning Curve for {name}")

plt.tight_layout()
plt.show()


# In[43]:


# Calibration curves   
plt.figure(figsize=(15, 20))

for idx, (name, model) in enumerate(ensemble_models.items(), 1):
    plt.subplot(len(ensemble_models), 1, idx)
    plot_calibration_curve(model, X_val, y_val, title=f"Calibration Curve for {name}")

plt.tight_layout()
plt.show()


# In[36]:


# Print Hypothesis
# Define model combinations for hypothesis comparisons
model_combinations = [('AB', 'RF'), ('AB', 'GB'), ('RF', 'GB')]

# Print hypothesis statements for each combination
for model1, model2 in model_combinations:
    print(f"Objective: To determine if {model1}'s performance is significantly different than {model2} in terms of accuracy.\n")
    
    print(f"H0: {model1}'s performance is equivalent to {model2}.")
    
    print(f"H1: {model1}'s performance is significantly different than {model2}.\n")
    print("-"*50)

# Paired t-tests and Effect Sizes
print("\nPaired t-tests and Effect Sizes:")
for i, model_1_name in enumerate(names_ensemble[:-1]):
    for model_2_name in names_ensemble[i+1:]:
        t_stat, p_value = perform_paired_t_test(ensemble_models[model_1_name], ensemble_models[model_2_name], X_val, y_val)
        y_pred1 = ensemble_models[model_1_name].predict(X_val)
        y_pred2 = ensemble_models[model_2_name].predict(X_val)
        errors_model_1 = [0 if true == pred else 1 for true, pred in zip(y_val, y_pred1)]
        errors_model_2 = [0 if true == pred else 1 for true, pred in zip(y_val, y_pred2)]
        effect_size = cohens_d(errors_model_1, errors_model_2)
        print(f"{model_1_name} vs {model_2_name}:")
        print("p-value:", p_value)
        print("Effect Size:", effect_size)
        print("-"*30)

# Conclusion
print("\nConclusion:")
print("In this analysis, ensemble classifiers were trained and their performance was evaluated using validation accuracy,\n cross-validation scores, bootstrapped confidence intervals, paired t-tests, and effect sizes. The results provide insights into the relative performance of the models and their statistical differences.")


# In[38]:


# Create a stacking classifier with the trained base models
estimators = [(name, model) for name, model in ensemble_models.items()]
stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train the stacking classifier
stack_clf.fit(X_train, y_train)

# Evaluate the stacking classifier on the validation data
stacked_acc = stack_clf.score(X_val, y_val)
print("Stacking Classifier Accuracy:", stacked_acc)


# In[37]:


print("Objective: To determine if a stacked model offers superior performance to \nthe best-performing individual ensemble model in terms of accuracy.")
print('\n')
print("H0: The stacked model's performance is equivalent to the best-performing \nindividual ensemble model.")
print('\n')
print("H1: The stacked model's performance is significantly better than the \nbest-performing individual ensemble model.")


# In[39]:


# Identify the ensemble model with the highest accuracy on the validation set
best_model_name = names_ensemble[np.argmax(results_ensemble)]
best_model_acc = np.max(results_ensemble)

print(f"Best Ensemble Model: {best_model_name} with Accuracy: {best_model_acc:.4f}")

# Paired t-test: Compare stacked model performance vs. best-performing ensemble model
t_stat, p_value = perform_paired_t_test(stack_clf, ensemble_models[best_model_name], X_val, y_val)

# Effect Size calculation
y_pred_best = ensemble_models[best_model_name].predict(X_val)
errors_best_model = [0 if true == pred else 1 for true, pred in zip(y_val, y_pred_best)]
y_pred_stacked = stack_clf.predict(X_val)
errors_stacked_model = [0 if true == pred else 1 for true, pred in zip(y_val, y_pred_stacked)]

effect_size = cohens_d(errors_best_model, errors_stacked_model)

print("\nStacked Model vs Best-performing Ensemble Model:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Effect size: {effect_size: .4f}")
# Assessing Statistical Significance
alpha = 0.01
if p_value < alpha:
    print(f"The performance difference between the stacked model and the {best_model_name} is statistically significant at the {alpha*100}% level.")
else:
    print(f"There's no significant performance difference between the stacked model and the {best_model_name} at the {alpha*100}% level.")
    
print("\nConclusion:")
print("In this analysis, ensemble classifiers and a stacked classifier\nwas trained. Performances were evaluated and statistically compared.\nThe results provide insights into whether stacking offers additional\nvalue compared to individual ensemble methods.")


# In[40]:


# Visualize the performance metrics
model_names = [best_model_name, "Stacked"]
accuracies = [best_model_acc, stacked_acc]

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=model_names, y=accuracies, ax=ax)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Model")
plt.title("Comparison: Stacked Model vs Best Ensemble Model")
plt.grid(axis="y")
plt.show()


# In[41]:


# Comparison: ROC and AUC curves
plt.figure(figsize=(10, 8))
plot_roc_curve(ensemble_models[best_model_name], X_val, y_val, best_model_name)
plot_roc_curve(stack_clf, X_val, y_val, "Stacking Classifier")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()


# In[44]:


# Create a dictionary with only the best model and the stacking classifier
models_to_plot = {
    best_model_name: ensemble_models[best_model_name],
    'Stacking Classifier': stack_clf
}

# Plotting the learning curves
plt.figure(figsize=(10, 8))

for idx, (name, model) in enumerate(models_to_plot.items(), 1):
    plt.subplot(len(models_to_plot), 1, idx)
    plot_learning_curve(model, X, y, title=f"Learning Curve for {name}")

plt.tight_layout()
plt.show()

# Calibration curves
plt.figure(figsize=(10, 8))

for idx, (name, model) in enumerate(models_to_plot.items(), 1):
    plt.subplot(len(models_to_plot), 1, idx)
    plot_calibration_curve(model, X_val, y_val, title=f"Calibration Curve for {name}")

plt.tight_layout()
plt.show()


# In[69]:


#Importing finished data with stacking classifier applied
# Save the trained stacking classifier to disk
joblib.dump(stack_clf, 'stacking_classifier.pkl')

# Use this to load it back
loaded_model = joblib.load('stacking_classifier.pkl')


# In[70]:


# Define paths
html_path = r"C:\Users\ljfit\Desktop\UCDPA-L.J._Fitzgerald\UCD Assignment (Random Forest Importance Values).html"
images_dir = 'extracted_images'

# Create directory for the images if it doesn't exist
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Read HTML content
with open(html_path, "r", encoding='utf-8') as f:
    content = f.read()

# Extract all images into img_data_list
img_data_list = re.findall(r'data:image/png;base64,([^\"]+)', content)

# Remove problematic files
img_data_list = [img_data for i, img_data in enumerate(img_data_list) if i not in [0, 15]]

# Save images
for i, img_data in enumerate(img_data_list, start=1):
    # Ensure padding
    padding = 4 - (len(img_data) % 4)
    img_data += '=' * padding

    try:
        decoded_img = base64.b64decode(img_data.encode('utf-8'))
        with open(f"{images_dir}/image_{i}.png", "wb") as f:
            f.write(decoded_img)
    except Exception as e:
        print(f"Failed to decode and save image_{i}.png due to: {e}")


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

train = pd.read_csv(r"C:\Users\marac\Downloads\train.csv")
test = pd.read_csv(r"C:\Users\marac\Downloads\test.csv")


# Combine datasets for preprocessing to ensure consistency
full_data = pd.concat([train.drop('Survived',axis=1),test])

# extract and group titles
## this turns a unique identifier column into something we can test for correlation with 
## survival. many titles appear to infrquently to garner their own category, group into "Rare"
full_data['Title'] = full_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_mapping = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Don": "Rare",
    "Lady": "Rare",
    "Countess": "Rare",
    "Jonkheer": "Rare",
    "Sir": "Rare",
    "Capt": "Rare",
    "Ms": "Miss",
    "Dona": "Rare"
}
full_data['Title'] = full_data['Title'].map(title_mapping)

# Create family size feature
## siblings on shipt + parents on ship + self
full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch'] + 1

# Create is_alone feature
full_data['IsAlone'] = (full_data['FamilySize'] == 1).astype(int)

# Create age bands
## bining the conttinuous variable age will let us capture non-linear effects and reduce noise
full_data['AgeBand'] = pd.cut(full_data['Age'], 5)

# Get the length of cabin string (proxy for wealth/status)
full_data['CabinBool'] = (full_data['Cabin'].notnull()).astype(int)

# Create fare per person
full_data['FarePerPerson'] = full_data['Fare'] / (full_data['FamilySize'])

full_data['age_class'] =full_data['Age'] *full_data['Pclass']

# Split back to train/test
train_processed = full_data.iloc[:len(train)].copy()
test_processed = full_data.iloc[len(train):].copy()

# Recombine with survival data for training
train_processed.loc[:, 'Survived'] = train['Survived']

# Define features to use
numerical_features = ['Age', 'Fare', 'FamilySize', 'FarePerPerson','age_class']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'CabinBool', 'IsAlone','AgeBand']

# Define preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models to try
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(random_state=42)
}
base_models = models.copy()

# Model Evaluation
X = train_processed[numerical_features + categorical_features]
y = train_processed['Survived']

# Split data for model evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
results = {}
for name, model in models.items():
    # Create and fit the full pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit the pipeline on training data
    full_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = full_pipeline.predict(X_val)
    
    # Evaluate
    accuracy = accuracy_score(y_val, y_pred)
    results[name] = accuracy
    
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_val, y_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(full_pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
    

# Get top 3 models for ensemble
top_models = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
top_model_names = [name for name, _ in top_models]
print(f"\nTop 3 models for ensemble: {', '.join(top_model_names)}")

# Create an ensemble (Voting Classifier) with the top 3 models
estimators = []
for name in top_model_names:
    estimators.append((name, base_models[name]))

# Add the voting classifier to our models
voting_classifier = VotingClassifier(estimators=estimators, voting='soft')
models['VotingEnsemble'] = voting_classifier

# Create and evaluate the ensemble pipeline
ensemble_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', voting_classifier)
])
# Fit on training data
ensemble_pipeline.fit(X_train, y_train)

# Make predictions
y_ensemble_pred = ensemble_pipeline.predict(X_val)

# Evaluate ensemble
ensemble_accuracy = accuracy_score(y_val, y_ensemble_pred)
results['VotingEnsemble'] = ensemble_accuracy

print(f"\nVoting Ensemble Accuracy: {ensemble_accuracy:.4f}")
print(f"\nClassification Report for Voting Ensemble:")
print(classification_report(y_val, y_ensemble_pred))

# Cross-validation score for ensemble
cv_scores = cross_val_score(ensemble_pipeline, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

# Choose the best model
best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Create full pipeline with best model
best_model = models[best_model_name]
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])


# Define models and their parameter grids for GridSearchCV
param_grid = {
    'RandomForest': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 5, 10],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'LogisticRegression': {
        'model__C': [0.001, 0.01, 0.1, 1, 10],
        'model__solver': ['liblinear', 'lbfgs'],  # Add solvers
        'model__penalty': ['l1', 'l2']  # Add penalty
    }
}

# Models to tune
models_to_tune = ['RandomForest', 'GradientBoosting', 'LogisticRegression']
best_models = {} # To store the best models

# Perform Grid Search for each model
for model_name in models_to_tune:
    print(f"\nTuning {model_name}...")
    model = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000) # set max_iter
    }[model_name]
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Use StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid Search
    grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    # Print best parameters and score
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation score for {model_name}: {grid_search.best_score_:.4f}")

# Evaluate the best models on the validation set
print("\nValidation set performance of the best models:")
for model_name, best_model in best_models.items():
    y_pred = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\n{model_name} Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_val, y_pred))


# Choose the best model
best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Create full pipeline with best model
best_model = models[best_model_name]
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])

# Feature Importance for tree-based models
if best_model_name in ['RandomForest', 'GradientBoosting']:
    # Get feature names after preprocessing
    feature_names = []
    
    # Get numerical feature names
    for feature in numerical_features:
        feature_names.append(feature)
    
    # Get one-hot encoded categorical feature names
    for i, feature in enumerate(categorical_features):
        categories = final_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].categories_[i]
        for category in categories:
            feature_names.append(f"{feature}_{category}")
    
    # Get feature importances
    importances = final_pipeline.named_steps['model'].feature_importances_
    
    # Create DataFrame for visualization
    feature_imp = pd.DataFrame({'Feature': feature_names[:len(importances)], 'Importance': importances})
    feature_imp = feature_imp.sort_values('Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp.head(15))
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    print("\nTop 10 important features:")
    print(feature_imp.head(10))
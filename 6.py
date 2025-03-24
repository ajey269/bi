logistic regression 

# Load necessary libraries
library(dplyr)
library(titanic)
library(pROC)

# Load Titanic dataset
titanic_train <- titanic::titanic_train  # Fix dataset loading

# Check dataset structure
head(titanic_train)

# Data Cleaning: Removing rows with missing values
titanic_clean <- titanic_train %>%
  filter(!is.na(Age), !is.na(Embarked), !is.na(Sex), !is.na(Pclass))

# Convert categorical variables to factors
titanic_clean$Survived <- as.factor(titanic_clean$Survived)
titanic_clean$Pclass <- as.factor(titanic_clean$Pclass)
titanic_clean$Sex <- as.factor(titanic_clean$Sex)
titanic_clean$Embarked <- as.factor(titanic_clean$Embarked)

# Build Logistic Regression Model
model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data = titanic_clean, family = binomial)

# Print model summary
summary(model)

# Predict probabilities
predictions <- predict(model, type = "response")

# Convert probabilities to binary classification
predictions_class <- ifelse(predictions > 0.5, 1, 0)

# Evaluate Model Accuracy
confusion_matrix <- table(Predicted = predictions_class, Actual = titanic_clean$Survived)
print(confusion_matrix)

# Correct accuracy calculation
accuracy <- mean(predictions_class == as.integer(as.character(titanic_clean$Survived)))
print(paste("Accuracy:", accuracy))

# ROC Curve (Fixing factor issue)
roc_curve <- roc(as.integer(as.character(titanic_clean$Survived)), predictions)
plot(roc_curve, main = "ROC Curve")

/////////////

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Load Titanic dataset from seaborn
titanic = sns.load_dataset("titanic")

# Select relevant columns and drop missing values
titanic_clean = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].dropna()

# Convert categorical variables to numerical
titanic_clean['sex'] = titanic_clean['sex'].map({'male': 0, 'female': 1})
titanic_clean['embarked'] = titanic_clean['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Define features (X) and target variable (y)
X = titanic_clean[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = titanic_clean['survived']

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Convert probabilities to binary classification (Threshold = 0.5)
y_pred = (y_pred_proba > 0.5).astype(int)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")  # Save the plot
plt.show()


pip install pandas numpy matplotlib scikit-learn seaborn

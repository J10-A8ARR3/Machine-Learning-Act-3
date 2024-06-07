import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Load labeled data
labeled_df = pd.read_csv(r'D:\BSCS 3-4\MACHINE LEARNING\ACT 3\ML - ACT 3\spam.csv')

# Load unlabeled data
unlabeled_df = pd.read_csv(r'D:\BSCS 3-4\MACHINE LEARNING\ACT 3\ML - ACT 3\unlabelspam\unlabelspam.csv')

# Turn spam/ham into numerical data in the labeled dataset
labeled_df['spam'] = labeled_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split labeled data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(labeled_df.Message, labeled_df.spam, test_size=0.25)

# Find the word count and store data as a matrix
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)

# Train Inductive Model
inductive_model = MultinomialNB()
inductive_model.fit(x_train_count, y_train)

# Test Inductive Model
x_test_count = cv.transform(x_test)
y_pred_inductive = inductive_model.predict(x_test_count)
inductive_report = classification_report(y_test, y_pred_inductive, output_dict=True)

# Calculate Inductive Model Accuracy
inductive_accuracy = accuracy_score(y_test, y_pred_inductive)

# Transductive Learning (Semi-Supervised)
# Use the inductive model to label the unlabeled data
unlabeled_count = cv.transform(unlabeled_df.Message)
unlabeled_predictions = inductive_model.predict(unlabeled_count)

# Combine the newly labeled data with the original training data
x_combined = np.concatenate([x_train_count.toarray(), unlabeled_count.toarray()])
y_combined = np.concatenate([y_train, unlabeled_predictions])

# Retrain the model on the combined dataset
transductive_model = MultinomialNB()
transductive_model.fit(x_combined, y_combined)

# Test Transductive Model
y_pred_transductive = transductive_model.predict(x_test_count)
transductive_report = classification_report(y_test, y_pred_transductive, output_dict=True)

# Calculate Transductive Model Accuracy
transductive_accuracy = accuracy_score(y_test, y_pred_transductive)

# Save the tagged unlabeled data to a new CSV file
unlabeled_df['Predicted_Category'] = unlabeled_predictions
unlabeled_df['Predicted_Category'] = unlabeled_df['Predicted_Category'].apply(lambda x: 'spam' if x == 1 else 'ham')

# Reorder columns
tagged_unlabeled_df = unlabeled_df[['Predicted_Category', 'Message']]

# Save to CSV
tagged_unlabeled_df.to_csv(r'D:\BSCS 3-4\MACHINE LEARNING\ACT 3\ML - ACT 3\unlabelspam\tagged_unlabeled_data.csv', index=False)

# Display results
print("Inductive Model Classification Report:")
print(classification_report(y_test, y_pred_inductive))

print("Transductive Model Classification Report:")
print(classification_report(y_test, y_pred_transductive))

print(f"Inductive Model Accuracy: {inductive_accuracy}")
print(f"Transductive Model Accuracy: {transductive_accuracy}")

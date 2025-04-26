# chunk_1_train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# 1. Load the dataset
df = pd.read_csv('GIB_part2.csv')

# 2. Quick inspection
print("Dataset shape:", df.shape)
print(df.columns.tolist())
print(df.head())

# 3. Drop rows with missing Disposition
df = df.dropna(subset=['Disposition'])
df.drop(['hematocrit_drop (if available)'], axis=1, inplace=True)
df['Platelets'].fillna(df['Platelets'].mean(), inplace=True)

# 4. Define features and target
feature_cols = [col for col in df.columns if col != 'Disposition']
X = df[feature_cols]
# Shift Disposition (1–3) to 0–2 for sklearn
y = df['Disposition'].astype(int) - 1

# 5. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 6. Train multinomial logistic regression
clf = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000,
    solver='lbfgs'
)
clf.fit(X_train, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test)
print("Classification report on test set:")
print(classification_report(y_test, y_pred, target_names=["Not ICU","ICU","Inpatient"]))

# 8. Save the trained model
with open('disposition_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model training complete and saved to disposition_model.pkl")

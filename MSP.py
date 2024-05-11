import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv')
if 'Unnamed: 0' in df.columns:                                                              # Dropping the first column "id" which is unnamed in this case
    df.drop('Unnamed: 0', axis=1, inplace=True)

categorical_vars = ['Breastfeeding', 'Varicella', 'Mono_or_Polysymptomatic', 'Gender']      # Applying one-hot encoding to categorical variables that are non-ordinal
df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)
df.drop(['Initial_EDSS', 'Final_EDSS'], axis=1, inplace=True)                               # Dropping due to high correlation and a large prectage of nulls.

df['Schooling'].fillna(df['Schooling'].median(), inplace=True)                              # Imputing missing values with median for numerical columns
df['Initial_Symptom'].fillna(df['Initial_Symptom'].median(), inplace=True)

Q1 = df.select_dtypes(include=['float64', 'int64']).quantile(0.25)                          # Detecting outliers using the IQR range and capping
Q3 = df.select_dtypes(include=['float64', 'int64']).quantile(0.75)
IQR = Q3 - Q1

exclude_columns = ['BAEP', 'VEP', 'LLSSEP', 'ULSSEP']
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    if column not in exclude_columns:
        lower_bound = Q1[column] - 1.5 * IQR[column]
        upper_bound = Q3[column] + 1.5 * IQR[column]
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])

y = df['group']
x = df[['Age', 'Schooling', 'Initial_Symptom', 'Oligoclonal_Bands', 'LLSSEP',
       'ULSSEP', 'VEP', 'BAEP', 'Periventricular_MRI', 'Cortical_MRI',
       'Infratentorial_MRI', 'Spinal_Cord_MRI', 'Breastfeeding_2',
       'Breastfeeding_3', 'Varicella_2', 'Varicella_3',
       'Mono_or_Polysymptomatic_2', 'Mono_or_Polysymptomatic_3', 'Gender_2']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

pickle.dump(rf_classifier,open('model.pkl','wb'))
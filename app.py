import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


df = pd.read_csv('test.csv')

print(df.head())


print(df.isnull().sum())


def clean_numeric_columns(column):
    return pd.to_numeric(df[column].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')


numeric_columns = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                   'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                   'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                   'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
                   'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

for col in numeric_columns:
    df[col] = clean_numeric_columns(col)

df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)


le = LabelEncoder()
df['Name'] = le.fit_transform(df['Name'])  
df['Occupation'] = le.fit_transform(df['Occupation'])
df['Type_of_Loan'] = le.fit_transform(df['Type_of_Loan'])
df['Payment_of_Min_Amount'] = le.fit_transform(df['Payment_of_Min_Amount'])
df['Payment_Behaviour'] = le.fit_transform(df['Payment_Behaviour'])


df = pd.get_dummies(df, columns=['Month'], drop_first=True)
X = df.drop(['ID', 'Customer_ID', 'SSN', 'Credit_Mix'], axis=1)  
y = df['Credit_Mix']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)




accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


output_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
output_df.to_csv('credit_score_predictions.csv', index=False)


feature_importances = model.feature_importances_
features = X.columns


feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))


with open('credit_scoring_model.pkl', 'wb') as file:
    pickle.dump(model, file)

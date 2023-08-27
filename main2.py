import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
excel_file_path = 'sample_container_data.xlsx'
df = pd.read_excel(excel_file_path)

# Prepare the data
X = df.drop(['location'], axis=1)
y = df['location']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict location for the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Create a DataFrame to store results
result_df = pd.DataFrame({'True_Location': y_test, 'Predicted_Location': y_pred})

# Save results to an Excel file
result_excel_file_path = 'result.xlsx'
result_df.to_excel(result_excel_file_path, index=False)

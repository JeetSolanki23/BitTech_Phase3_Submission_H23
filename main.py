import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the Excel file
excel_file_path = "Data.xlsx"

# Load Yard Locations Data
yard_locations_data = pd.read_excel(excel_file_path, sheet_name="Yard Locations")

# Load Past In and Out Data
past_in_out_data = pd.read_excel(excel_file_path, sheet_name="Past In and Out Container Data")

# Load Incoming Container Data
incoming_container_data = pd.read_excel(excel_file_path, sheet_name="Incoming Conatiners")

# Filter and select necessary columns for machine learning predictions
selected_columns_yard = ["Container Size", "Location", "Area", "Row", "Bay", "Level"]
filtered_yard_data = yard_locations_data[selected_columns_yard]

selected_columns_past = ["CON_NUM", "IN_TIME", "OUT_TIME", "VALIDITY"]
filtered_past_in_out_data = past_in_out_data[selected_columns_past]

selected_columns_incoming = ["ID", "IN_TIME", "STATUS"]
filtered_incoming_container_data = incoming_container_data[selected_columns_incoming]

# Merge filtered dataframes
merged_data = pd.merge(filtered_incoming_container_data, filtered_past_in_out_data, on="ID")

# Prepare features and target variable
X = merged_data[["ID", "IMPORT_EXPORT", "CON_SIZE"]]  # Features: ID, IMPORT_EXPORT, and CON_SIZE
y = merged_data["Location"]  # Target variable: Assigned Location

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Now, let's predict the location based on the given attributes
input_data = pd.DataFrame({
    "ID": [123],               # Example: Container ID
    "IMPORT_EXPORT": [2],      # Example: 2 corresponds to Export
    "CON_SIZE": [20]           # Example: 20 ft container size
})

# Use the trained model to predict the location
predicted_location = model.predict(input_data)

print("Predicted Location:", predicted_location)

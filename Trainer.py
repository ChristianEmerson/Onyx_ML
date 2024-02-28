import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv("Trainer.csv")

# Preprocessing
# One-hot encode the "category" column

data.replace({"category":{'gain':0,'loss':1,'fit':2}}, inplace=True)

# Split data into features (X) and target variables (y)
X = data.drop(columns=["set", "reps", "weights", "Duration"])
y = data[["set", "reps", "weights", "Duration"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)


# Evaluate the model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', marker='o', label='Actual vs. Predicted')

# Add labels and title
plt.xlabel('category,sleep,recovery,body')
plt.ylabel('set,reps,weights,Duration')
plt.title('category,sleep,recovery,body vs. set,reps,weights,Duration')

# Add a diagonal line for reference (perfect prediction)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')

# Add a legend
plt.legend()

# Show the plot
plt.show()
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

pickle.dump(model,open('Trainer.pkl','wb'))
Trainer=pickle.load(open('Trainer.pkl','rb'))
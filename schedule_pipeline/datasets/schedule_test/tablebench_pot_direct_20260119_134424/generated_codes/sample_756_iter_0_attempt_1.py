import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['registered voters'] = pd.to_numeric(df['registered voters'])
df['total candidates'] = pd.to_numeric(df['total candidates'])

# Prepare features (X) and target (y)
X = df[['registered voters']]
y = df['total candidates']

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for a new district with 500,000 registered voters
new_voters = [[500000]]
predicted_candidates = model.predict(new_voters)[0]

# Round to nearest whole number since candidates are integers
final_prediction = round(predicted_candidates)

print(f"Final Answer: {final_prediction}")
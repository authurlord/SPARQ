import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'viewers (in millions)' to float
df['viewers (in millions)'] = df['viewers (in millions)'].astype(float)

# Prepare the data for linear regression
X = np.arange(len(df)).reshape(-1, 1)  # Episode number as independent variable
y = df['viewers (in millions)'].values  # Viewership as dependent variable

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the viewership for the next episode (episode 7)
next_episode = len(df)
predicted_viewers = model.predict([[next_episode]])[0]

print(f"Final Answer: {predicted_viewers:.2f}")
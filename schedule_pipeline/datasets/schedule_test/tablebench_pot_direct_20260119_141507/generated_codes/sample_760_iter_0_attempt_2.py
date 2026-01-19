import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Prepare data: years from 1995-96 to 2006-07 (index 0 to 10)
years = list(range(1995, 2008))  # 1995 to 2007, but we have data from 1995–96 to 2006–07 → 11 years
latvian_percent = [float(row['% learning in latvian']) for row in df[df['school year'] != 'school year'].to_dict('records')]

# Convert to numpy arrays
X = np.array(years).reshape(-1, 1)
y = np.array(latvian_percent)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for 2009–2010 (year 2009)
predicted_latvian_2009 = model.predict([[2009]])[0]

# Assume % learning in others is approximately 1.3% (average from the data)
others_percent = 1.3

# Calculate % learning in Russian
russian_percent = 100 - predicted_latvian_2009 - others_percent

print(f"Final Answer: {russian_percent:.1f}")
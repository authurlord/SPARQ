import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Extract relevant columns
years = [1995, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006]
russian_percent = [
    (132540 / 337660) * 100,
    (120925 / 361432) * 100,
    (116009 / 359818) * 100,
    (108454 / 351989) * 100,
    (101486 / 340308) * 100,
    (95841 / 327358) * 100,
    (84559 / 300667) * 100,
    (77471 / 283947) * 100,
    (70683 / 266111) * 100
]

# Convert to DataFrame
X = np.array(years).reshape(-1, 1)
y = np.array(russian_percent)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for year 2009
predicted_russian_percent_2009 = model.predict([[2009]])[0]

print(f"Final Answer: {predicted_russian_percent_2009:.1f}")
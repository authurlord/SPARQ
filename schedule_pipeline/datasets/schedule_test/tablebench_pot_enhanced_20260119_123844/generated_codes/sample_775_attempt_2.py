import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert success rate to float by removing '%' and converting to decimal
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)

# Extract years and success rates
years = df['year'].astype(int)
success_rates = df['success rate']

# Fit a linear regression model
X = years.values.reshape(-1, 1)
y = success_rates.values
model = np.polyfit(X.flatten(), y, 1)
predicted_rates = np.polyval(model, np.arange(2014, 2019))

# Print the forecasted success rates
forecasted_rates = [f"{rate:.2f}" for rate in predicted_rates]
print(f"Final Answer: {', '.join(forecasted_rates)}")
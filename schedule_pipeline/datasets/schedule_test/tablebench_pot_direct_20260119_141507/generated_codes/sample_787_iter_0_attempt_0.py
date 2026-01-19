import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert all values to float
data = df.values.astype(float)

# Define years
years = ['2006', '2007', '2008', '2009', '2010']

# Initialize variables
min_std_dev = float('inf')
best_category_idx = 0
best_cagr = 0

# For each row (category)
for i in range(data.shape[0]):
    values = data[i]
    if values[0] == 0:
        continue  # Skip if initial value is zero
    
    # Compute annual growth rates (percentage change)
    growth_rates = []
    for j in range(1, len(values)):
        if values[j-1] == 0:
            growth_rates.append(np.nan)
        else:
            growth_rate = (values[j] / values[j-1]) - 1
            growth_rates.append(growth_rate)
    
    # Remove NaNs and compute standard deviation
    valid_growth_rates = [r for r in growth_rates if not np.isnan(r)]
    if len(valid_growth_rates) < 1:
        continue
    
    std_dev = np.std(valid_growth_rates)
    
    # If this is more consistent (lower std), update
    if std_dev < min_std_dev:
        min_std_dev = std_dev
        best_category_idx = i
        # Compute CAGR
        cagr = (values[-1] / values[0]) ** (1/4) - 1
        best_cagr = cagr * 100

# Output the category with most consistent growth and its average annual percentage increase
print(f"Final Answer: {best_cagr:.2f}%")
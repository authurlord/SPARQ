import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert all values in the data to float
data = df.values.astype(float)

# Extract column names (years)
years = df.columns.tolist()

# Define a function to compute average annual growth rate
def calculate_avg_annual_growth(row):
    values = row[1:]  # Exclude first year (2006) as base
    initial = values[0]
    final = values[-1]
    if initial == 0:
        return 0
    # Compound annual growth rate
    growth_rate = (final / initial) ** (1/4) - 1
    return growth_rate * 100  # Convert to percentage

# Compute average annual growth rate for each category
growth_rates = []
for i in range(data.shape[0]):
    growth_rate = calculate_avg_annual_growth(data[i])
    growth_rates.append(growth_rate)

# Find the category with the most consistent growth (smallest variance in annual growth rates)
# First, compute annual growth rates for each year
annual_growth_rates = []
for i in range(data.shape[0]):
    values = data[i][1:]  # Skip 2006
    if values[0] == 0:
        annual_growth_rates.append([np.nan] * 4)
        continue
    # Compute annual growth between consecutive years
    annual_rates = []
    for j in range(4):
        if values[j] == 0 or values[j+1] == 0:
            annual_rates.append(0)
        else:
            rate = ((values[j+1] / values[j]) - 1) * 100
            annual_rates.append(rate)
    annual_growth_rates.append(annual_rates)

# Compute variance of annual growth rates for each category
variances = []
for rates in annual_growth_rates:
    # Filter out NaNs
    valid_rates = [r for r in rates if not np.isnan(r)]
    if len(valid_rates) == 0:
        variances.append(np.inf)
    else:
        variance = np.var(valid_rates)
        variances.append(variance)

# Find the index with minimum variance (most consistent)
min_variance_idx = np.argmin(variances)
consistent_category = df.index[min_variance_idx]
avg_growth_rate = np.mean(annual_growth_rates[min_variance_idx])

# Final answer: category name and average annual percentage increase
print(f"Final Answer: {consistent_category}, {avg_growth_rate:.2f}")
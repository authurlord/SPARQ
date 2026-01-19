import pandas as pd

df = pd.read_csv('table.csv')
# Extract the columns of interest
capacity = df['commissioned capacity (mw)']
year_commission = df['year of commission']

# Calculate the correlation coefficient
correlation_coefficient = capacity.corr(year_commission)
print(f"Final Answer: {correlation_coefficient:.3f}")
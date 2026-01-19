import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert all columns to numeric
df = df.apply(pd.to_numeric)

# Calculate CAGR for each row: (final / initial)^(1/4) - 1
# Years: 2006 to 2010 → 4 intervals
initial = df.iloc[:, 0]  # 2006
final = df.iloc[:, -1]   # 2010
n = 4  # number of years

# Avoid division by zero or negative values
cagr = (final / initial) ** (1/n) - 1

# Find the row with the highest CAGR
max_cagr_idx = cagr.idxmax()
max_cagr_value = cagr.max()

# Since no category names are provided, we use the row index as the category name
category_name = f"Category {max_cagr_idx}"

# Output the result
print(f"Final Answer: {category_name}, {max_cagr_value:.3f}")
import pandas as pd

# Load the table data
df = pd.read_csv('table.csv')

# Convert all values to float
df = df.astype(float)

# Initialize variables to track the best category
best_category_idx = -1
min_std = float('inf')

# Iterate over each row (category)
for idx in df.index:
    row = df.loc[idx]
    # Calculate annual percentage changes
    pct_changes = []
    for i in range(1, len(row)):
        change = (row.iloc[i] - row.iloc[i-1]) / row.iloc[i-1] * 100
        pct_changes.append(change)
    
    # Compute standard deviation of changes
    std = pd.Series(pct_changes).std()
    
    # Update if this category has lower standard deviation (more consistent)
    if std < min_std:
        min_std = std
        best_category_idx = idx

# Get the average annual percentage increase for the most consistent category
best_row = df.loc[best_category_idx]
pct_changes = []
for i in range(1, len(best_row)):
    change = (best_row.iloc[i] - best_row.iloc[i-1]) / best_row.iloc[i-1] * 100
    pct_changes.append(change)

avg_annual_increase = pd.Series(pct_changes).mean()

# Output the result
print(f"Final Answer: {best_category_idx}, {avg_annual_increase:.2f}")
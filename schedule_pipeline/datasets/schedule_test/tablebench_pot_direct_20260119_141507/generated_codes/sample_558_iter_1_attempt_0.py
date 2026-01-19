import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows from 1935 to 1943 inclusive
filtered_df = df[df['Year'].between(1935, 1943)]

# Convert 'Quantity withdrawn' to numeric, handling any potential parsing issues
# The column contains strings like '1', '3', etc., so we can use pd.to_numeric with errors='coerce'
withdrawn_values = pd.to_numeric(filtered_df['Quantity withdrawn'], errors='coerce')

# Drop any invalid entries (if any)
withdrawn_values = withdrawn_values.dropna()

# Calculate average
average_withdrawn = withdrawn_values.mean()

print(f"Final Answer: {average_withdrawn:.1f}")
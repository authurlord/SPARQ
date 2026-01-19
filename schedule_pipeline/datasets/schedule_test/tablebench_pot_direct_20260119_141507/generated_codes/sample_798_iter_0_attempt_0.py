import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer and 'Conservative councillors' to integer
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Conservative councillors'] = pd.to_numeric(df['Conservative councillors'], errors='coerce')

# Filter years from 1947 to 1972 inclusive
filtered_df = df[(df['Year'] >= 1947) & (df['Year'] <= 1972)]

# Sort by year
filtered_df = filtered_df.sort_values('Year')

# Extract Conservative councillors values
conservative_values = filtered_df['Conservative councillors'].values

# Compute annual changes (difference between consecutive years)
changes = [conservative_values[i] - conservative_values[i-1] for i in range(1, len(conservative_values))]

# Average annual change
average_change = sum(changes) / len(changes) if changes else 0

print(f"Final Answer: {average_change:.2f}")
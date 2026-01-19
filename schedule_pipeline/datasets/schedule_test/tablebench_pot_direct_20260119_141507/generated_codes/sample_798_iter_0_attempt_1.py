import pandas as pd

df = pd.read_csv('table.csv')
# Convert Year to integer and extract Conservative councillors
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Conservative councillors'] = pd.to_numeric(df['Conservative councillors'], errors='coerce')

# Filter years from 1947 to 1972 inclusive
filtered_df = df[(df['Year'] >= 1947) & (df['Year'] <= 1972)]

# Sort by Year
filtered_df = filtered_df.sort_values('Year')

# Calculate year-over-year change in Conservative councillors
changes = filtered_df['Conservative councillors'].diff().dropna()

# Compute average change
average_change = changes.mean()

print(f"Final Answer: {average_change:.2f}")
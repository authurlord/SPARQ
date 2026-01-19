import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows from 1935 to 1943 inclusive
filtered_df = df[(df['Year'].astype(str).str.startswith('193')) | 
                  (df['Year'].astype(str).str.startswith('194'))]

# Extract the 'Quantity withdrawn' column and compute the mean
# Convert to numeric safely
withdrawn_values = pd.to_numeric(df['Quantity withdrawn'], errors='coerce')
filtered_withdrawn = withdrawn_values[(df['Year'] >= 1935) & (df['Year'] <= 1943)]

# Compute average
average_withdrawn = filtered_withdrawn.mean()

print(f"Final Answer: {average_withdrawn:.1f}")
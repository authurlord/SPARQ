import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows from 1935 to 1943 inclusive
filtered_df = df[(df['Year'].astype(str).str.startswith('193')) & (df['Year'].astype(str).str.endswith('5') | df['Year'].astype(str).str.endswith('6') | df['Year'].astype(str).str.endswith('7') | df['Year'].astype(str).str.endswith('8') | df['Year'].astype(str).str.endswith('9') | df['Year'].astype(str).str.endswith('4') | df['Year'].astype(str).str.endswith('3'))]

# Actually, better to filter by year directly
filtered_df = df[df['Year'].between(1935, 1943)]
# Convert 'Quantity withdrawn' to numeric, handling any non-numeric entries
filtered_df['Quantity withdrawn'] = pd.to_numeric(filtered_df['Quantity withdrawn'], errors='coerce')
# Drop NaN if any (in case of malformed data)
filtered_df = filtered_df.dropna(subset=['Quantity withdrawn'])
# Calculate average
average_withdrawn = filtered_df['Quantity withdrawn'].mean()
print(f"Final Answer: {average_withdrawn:.1f}")
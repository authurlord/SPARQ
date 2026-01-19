import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Confluence [by Lahn-km]' to numeric and filter values below 100
filtered_df = df[df['Confluence [by Lahn-km]'].astype(float) < 100]

# Convert 'Length [km]' to numeric, coercing errors to NaN
filtered_df['Length [km]'] = pd.to_numeric(filtered_df['Length [km]'], errors='coerce')

# Drop rows where Length is NaN
filtered_df = filtered_df.dropna(subset=['Length [km]'])

# Calculate average length
average_length = filtered_df['Length [km]'].mean()

print(f"Final Answer: {average_length:.1f}")
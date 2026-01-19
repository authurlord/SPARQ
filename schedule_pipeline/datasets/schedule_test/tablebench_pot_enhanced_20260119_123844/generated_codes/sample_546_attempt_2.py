import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total s ton' to numeric
df['total s ton'] = pd.to_numeric(df['total s ton'])

# Calculate the year-on-year increase
df['increase'] = df['total s ton'].diff()

# Find the row with the highest increase
max_increase_row = df.loc[df['increase'].idxmax()]

# Get the 'total s ton' value corresponding to the highest increase
final_value = max_increase_row['total s ton']
print(f"Final Answer: {final_value}")
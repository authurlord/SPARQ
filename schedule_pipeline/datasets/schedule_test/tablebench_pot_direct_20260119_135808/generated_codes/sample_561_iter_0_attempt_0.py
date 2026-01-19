import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1942 to 1946
df_filtered = df[df['Year'].astype(int).between(1942, 1946)]

# Extract and clean 'US Chart position' column
# Remove non-numeric characters from the string and convert to float
df_filtered['US Chart position'] = df_filtered['US Chart position'].str.replace(r'[^(0-9)]', '', regex=True)
df_filtered['US Chart position'] = pd.to_numeric(df_filtered['US Chart position'], errors='coerce')

# Calculate average, ignoring NaN values
average_position = df_filtered['US Chart position'].mean()

print(f"Final Answer: {average_position:.1f}")
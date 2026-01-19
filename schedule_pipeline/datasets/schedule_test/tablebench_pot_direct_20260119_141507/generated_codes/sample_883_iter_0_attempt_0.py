import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter data for 2001
df_2001 = df[df['country'].str.contains('2001', na=False)]

# Extract the 'aids orphans as % of orphans' column and find the maximum
max_percentage = df_2001['aids orphans as % of orphans'].max()
country_with_max = df_2001[df_2001['aids orphans as % of orphans'] == max_percentage]['country'].values[0]

# Get Uganda's percentage in 2001
uganda_2001_percentage = df[df['country'] == 'uganda (2001)']['aids orphans as % of orphans'].values[0]

print(f"Final Answer: {country_with_max}, {uganda_2001_percentage}")
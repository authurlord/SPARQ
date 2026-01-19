import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display the structure and first few rows
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Sort by date of polling to observe trends over time
df['date of polling'] = pd.to_datetime(df['date of polling'], errors='coerce')
df_sorted = df.sort_values('date of polling')

# Compute average support for each party over time
avg_support = df_sorted[['progressive conservative', 'liberal', 'new democratic']].mean()
print("\nAverage support across all polls (by party):")
print(avg_support)

# Observations: 
# - Progressive Conservative starts at 71% in early 2008 and rises to 82% in 2007, then dips slightly.
# - Liberal support is generally low (12–22%) and shows a slight increase from 2007 to 2009.
# - New Democratic Party support is relatively stable but slightly declines over time.

# Final Answer: The table contains polling data on three political parties across multiple dates. The Progressive Conservative party had the highest average support (about 71%), followed by New Democratic (about 18%), and Liberal (about 17%). Support for the Progressive Conservative party fluctuated slightly, peaking in early 2007 and declining slightly later, while Liberal support remained modest and stable. New Democratic support was consistently low and showed minor fluctuations.
Final Answer: Progressive Conservative, Liberal, New Democratic
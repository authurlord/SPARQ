import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display column names and first few rows for understanding
print("Columns:", df.columns.tolist())

# Convert 'date of polling' to datetime for better analysis
df['date of polling'] = pd.to_datetime(df['date of polling'], errors='coerce')

# Sort by date to observe trends over time
df_sorted = df.sort_values(by='date of polling')

# Group by date and compute average support for each party
party_support_avg = df_sorted.groupby('date of polling')[['progressive conservative', 'liberal', 'new democratic']].mean()

# Display the average support over time
print("\nAverage support per party over time:")
print(party_support_avg)

# Observations: 
# - Progressive Conservative support starts at ~72% in 2007, peaks at 82% in 2007 (Nov), then declines to ~59% in 2011.
# - Liberal support is relatively stable, ranging from 12% to 22%, with a slight increase from 2007 to 2011.
# - New Democratic support ranges from 5% to 33%, generally lower than others.

# Final Answer: The table includes polling data on political party support over time. Progressive Conservative support peaked in 2007 and declined, Liberal support remained relatively stable, and New Democratic support varied but was generally low. 
Final Answer: Progressive Conservative, Liberal, New Democratic
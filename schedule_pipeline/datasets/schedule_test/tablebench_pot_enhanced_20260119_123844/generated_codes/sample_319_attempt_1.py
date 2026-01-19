import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Stockholm Marathon and Marathon event
stockholm_marathon = df[(df['Competition'] == 'Stockholm Marathon') & (df['Event'] == 'Marathon')]
# Filter for times less than 2:35:00
# Since times are in format HH:MM:SS, string comparison works
fast_times = stockholm_marathon[stockholm_marathon['Notes'] < '2:35:00']
# Extract the year
years = fast_times['Year'].tolist()
print(f"Final Answer: {','.join(years)}")
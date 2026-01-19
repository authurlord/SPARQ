import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Stockholm Marathon and time less than 2:35:00
stockholm_marathon = df[(df['Event'] == 'Marathon') & (df['Venue'] == 'Stockholm, Sweden')]
# Filter by time < 2:35:00
fast_times = stockholm_marathon[stockholm_marathon['Notes'] < '2:35:00']
# Extract the year
year = fast_times['Year'].iloc[0]
print(f"Final Answer: {year}")
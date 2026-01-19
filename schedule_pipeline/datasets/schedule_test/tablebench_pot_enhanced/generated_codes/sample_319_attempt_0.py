import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Stockholm Marathon and Marathon event
stockholm_marathons = df[(df['Competition'] == 'Stockholm Marathon') & (df['Event'] == 'Marathon')]
# Convert time in Notes to timedelta
stockholm_marathons['time'] = pd.to_timedelta(stockholm_marathons['Notes'])
# Filter for times less than 2:35:00
fast_times = stockholm_marathons[stockholm_marathons['time'] < pd.Timedelta(hours=2, minutes=35)]
# Extract the year
year = fast_times['Year'].values[0]
print(f"Final Answer: {year}")
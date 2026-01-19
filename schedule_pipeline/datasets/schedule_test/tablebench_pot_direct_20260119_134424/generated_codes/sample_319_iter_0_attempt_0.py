import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Stockholm Marathon and Marathon event
stockholm_marathons = df[(df['Competition'] == 'Stockholm Marathon') & (df['Event'] == 'Marathon')]
# Convert Notes (time) to timedelta
stockholm_marathons['Notes'] = pd.to_timedelta(stockholm_marathons['Notes'])
# Filter times less than 2:35:00
fast_times = stockholm_marathons[stockholm_marathons['Notes'] < pd.Timedelta('2:35:00')]
# Get the year
year = fast_times['Year'].iloc[0]
print(f"Final Answer: {year}")
import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Stockholm Marathon and Marathon event
stockholm_marathon = df[(df['Competition'] == 'Stockholm Marathon') & (df['Event'] == 'Marathon')]
# Convert 'Notes' to timedelta for time comparison
stockholm_marathon['time'] = pd.to_timedelta(stockholm_marathon['Notes'])
# Filter for times less than 2:35:00
fast_times = stockholm_marathon[stockholm_marathon['time'] < pd.Timedelta('2:35:00')]
# Extract the year
years = fast_times['Year'].tolist()
print(f"Final Answer: {', '.join(years)}")
import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Stockholm Marathon and Marathon event
stockholm_marathons = df[(df['Competition'] == 'Stockholm Marathon') & (df['Event'] == 'Marathon')]
# Convert Notes (time) to timedelta
stockholm_marathons['Notes'] = pd.to_timedelta(stockholm_marathons['Notes'], errors='coerce')
# Filter for times less than 2:35:00
fast_times = stockholm_marathons[stockholm_marathons['Notes'] < pd.Timedelta('2:35:00')]
# Get the year(s)
years = fast_times['Year'].tolist()
print(f"Final Answer: {', '.join(years)}")
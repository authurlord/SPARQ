import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where venue is Stockholm, Sweden and event is Marathon
filtered = df[(df['Venue'] == 'Stockholm, Sweden') & (df['Event'] == 'Marathon')]

# Convert time to seconds for comparison
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# Apply condition: time less than 2:35:00 (i.e., < 2*3600 + 35*60 + 0)
valid_times = filtered[filtered['Notes'].apply(lambda x: x.startswith('2:') and len(x.split(':')) == 3)]
valid_times = valid_times[valid_times['Notes'].apply(lambda x: time_to_seconds(x) < 2*3600 + 35*60)]

# Extract the year (first row's Year, since it's the only one per year in this case)
years = valid_times['Year'].dropna().tolist()

# Since the question asks for "the year", and there might be multiple, we list them
print(f"Final Answer: {', '.join(years)}")
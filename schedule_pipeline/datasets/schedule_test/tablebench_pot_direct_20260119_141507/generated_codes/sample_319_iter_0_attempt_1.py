import pandas as pd

df = pd.read_csv('table.csv')

# Filter for Stockholm Marathon events
stockholm_marathon = df[(df['Venue'] == 'Stockholm, Sweden') & (df['Event'] == 'Marathon')]

# Convert time string to seconds for comparison
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# Filter times less than 2:35:00 (i.e., less than 2*3600 + 35*60 + 0)
threshold_seconds = 2 * 3600 + 35 * 60
filtered_rows = stockholm_marathon[stockholm_marathon['Notes'].apply(lambda x: time_to_seconds(x) < threshold_seconds)]

# Extract the year
years = filtered_rows['Year'].tolist()
print(f"Final Answer: {years}")
import pandas as pd

df = pd.read_csv('table.csv')
# Extract live births for the specified periods
relevant_periods = ['1950-1955', '1955-1960', '1960-1965', '1965-1970', '1970-1975', '1975-1980']
live_births_values = df[df['Period'].isin(relevant_periods)]['Live births per year']

# Clean and convert to integers
cleaned_values = []
for val in live_births_values:
    cleaned_val = int(val.replace(' ', ''))
    cleaned_values.append(cleaned_val)

total_live_births = sum(cleaned_values)
print(f"Final Answer: {total_live_births}")
import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Extract year from each date string in the 'june 10 - 11', 'march 27 - 29', etc. columns
years = []
for col in df.columns:
    for date_str in df[col]:
        if isinstance(date_str, str):
            year_part = date_str.split(', ')[-1]
            try:
                year = int(year_part)
                if year >= 1990:
                    years.append(year)
            except ValueError:
                continue

# Count the number of valid years >= 1990
count = len(years)
print(f"Final Answer: {count}")
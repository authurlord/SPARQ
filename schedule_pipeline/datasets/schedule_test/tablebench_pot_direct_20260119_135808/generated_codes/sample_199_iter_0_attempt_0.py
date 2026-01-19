import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'deaths' column
def clean_deaths(death_str):
    if death_str == 'not known' or death_str == 'none':
        return 0
    elif '+' in death_str:
        return int(death_str.replace('+', ''))  # Assume it's at least that value
    else:
        return int(death_str)

df['deaths'] = df['deaths'].apply(clean_deaths)

# Convert to numeric
df['number of major hurricanes'] = pd.to_numeric(df['number of major hurricanes'])
df['deaths'] = pd.to_numeric(df['deaths'])

# Calculate correlation
correlation = df['number of major hurricanes'].corr(df['deaths'])

# Output result
if correlation > 0:
    print("Final Answer: Yes")
else:
    print("Final Answer: No")
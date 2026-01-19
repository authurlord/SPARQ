import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'deaths' to numeric, handling special cases
def parse_deaths(death_str):
    if death_str == 'not known':
        return None
    elif death_str == 'none':
        return 0
    elif '+' in death_str:
        return int(death_str.replace('+', ''))  # Assume minimum value
    else:
        return int(death_str)

df['deaths'] = df['deaths'].apply(parse_deaths)

# Drop rows with missing deaths
df_clean = df.dropna(subset=['deaths'])

# Calculate correlation between major hurricanes and deaths
correlation = df_clean['number of major hurricanes'].corr(df_clean['deaths'])

print(f"Final Answer: {correlation:.2f}")
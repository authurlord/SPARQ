import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'deaths' column
def clean_deaths(death_str):
    if death_str == 'not known' or death_str == 'none':
        return 0
    elif '+' in death_str:
        return int(death_str.replace('+', ''))  # Assume the value is the base number
    else:
        return int(death_str)

df['deaths'] = df['deaths'].apply(clean_deaths)

# Calculate correlation between major hurricanes and deaths
correlation = df['number of major hurricanes'].astype(int).corr(df['deaths'])

print(f"Final Answer: {correlation:.2f}")
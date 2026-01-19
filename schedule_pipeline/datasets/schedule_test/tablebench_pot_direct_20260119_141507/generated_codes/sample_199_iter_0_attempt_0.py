import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert "deaths" to numeric, handling special cases
def parse_deaths(death_str):
    if pd.isna(death_str) or death_str == 'not known':
        return np.nan
    elif death_str == '100 +':
        return 100
    elif death_str == '30 +':
        return 30
    elif death_str == '200 +':
        return 200
    elif death_str == 'none':
        return 0
    else:
        return int(death_str)

df['deaths_numeric'] = df['deaths'].apply(parse_deaths)

# Drop rows where deaths are missing
df_filtered = df.dropna(subset=['deaths_numeric'])

# Calculate correlation between number of major hurricanes and deaths
correlation = df_filtered['number of major hurricanes'].corr(df_filtered['deaths_numeric'])

print(f"Final Answer: {correlation:.2f}")
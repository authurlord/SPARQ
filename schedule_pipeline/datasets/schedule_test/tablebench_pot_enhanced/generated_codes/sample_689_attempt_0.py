import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'years of kindergarten legally required' to numeric, handling special cases
def parse_years(year_str):
    if isinstance(year_str, str):
        if 'or' in year_str:
            # Extract the higher number in cases like '0 or 2'
            return max(int(x.strip()) for x in year_str.split('or'))
        else:
            return int(year_str)
    return year_str

df['years of kindergarten legally required'] = df['years of kindergarten legally required'].apply(parse_years)

# Find the canton with the maximum value
max_canton = df.loc[df['years of kindergarten legally required'].idxmax(), 'canton']
print(f"Final Answer: {max_canton}")
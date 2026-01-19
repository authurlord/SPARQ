import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'deaths' to numeric values
def convert_deaths(death_str):
    if death_str == 'not known' or death_str == 'none':
        return 0
    elif death_str == '100 +':
        return 100
    elif death_str == '30 +':
        return 30
    elif death_str == '200 +':
        return 200
    elif death_str == 'four & six':
        return 0
    else:
        try:
            return int(death_str)
        except:
            return 0

df['deaths'] = df['deaths'].apply(convert_deaths)

# Calculate correlation between major hurricanes and deaths
correlation = df['number of major hurricanes'].astype(int).corr(df['deaths'])

# Print result
if correlation > 0:
    print("Final Answer: yes")
else:
    print("Final Answer: no")
import pandas as pd

df = pd.read_csv('table.csv')

# Check for negative growth in average absolute annual growth
negative_growth_countries = df[df['average absolute annual growth'] < 0]

# Also check for extreme values in relative growth or population share
# But the most obvious deviation is Jordan due to negative growth

# Filter and inspect
deviant_country = negative_growth_countries['country (or dependent territory)'].values[0] if not negative_growth_countries.empty else "none"

print(f"Final Answer: {deviant_country}")
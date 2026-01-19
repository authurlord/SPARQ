import pandas as pd

df = pd.read_csv('table.csv')
# Find the canton with the highest number of years of kindergarten legally required
max_years_required = df['years of kindergarten legally required'].max()
canton_with_max_years = df[df['years of kindergarten legally required'] == max_years_required]['canton'].values[0]
print(f"Final Answer: {canton_with_max_years}")
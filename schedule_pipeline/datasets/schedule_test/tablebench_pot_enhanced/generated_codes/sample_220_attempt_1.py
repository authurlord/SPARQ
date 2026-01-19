import pandas as pd

df = pd.read_csv('table.csv')
# Filter data between 1959 and 1989
filtered_df = df[(df['year (january)'] >= '1959') & (df['year (january)'] <= '1989')]
# Convert population to numeric for analysis
filtered_df['population (000)'] = pd.to_numeric(filtered_df['population (000)'])
# Compute correlation between population and urban percentage
correlation = filtered_df['population (000)'].corr(pd.to_numeric(filtered_df['urban , %']))
print(f"Final Answer: {correlation:.2f}")
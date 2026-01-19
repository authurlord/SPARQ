import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for 'j17' and 'e4' classes
filtered_df = df[(df['class'] == 'j17') | (df['class'] == 'e4')]

# Calculate total number of locomotives at Doncaster and Pyewipe
total_locomotives = filtered_df['number at doncaster'].sum() + filtered_df['number at pyewipe'].sum()

print(f"Final Answer: {total_locomotives}")
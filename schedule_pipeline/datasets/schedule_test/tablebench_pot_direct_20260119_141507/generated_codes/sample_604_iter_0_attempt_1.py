import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for 'j17' and 'e4' classes
filtered_rows = df[(df['class'] == 'j17') | (df['class'] == 'e4')]

# Calculate total number of locomotives at Doncaster and Pyewipe
total_locomotives = (
    filtered_rows['number at doncaster'].sum() +
    filtered_rows['number at pyewipe'].sum()
)

print(f"Final Answer: {total_locomotives}")
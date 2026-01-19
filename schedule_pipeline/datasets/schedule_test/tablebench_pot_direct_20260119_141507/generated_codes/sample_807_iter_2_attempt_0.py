import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Division is 'Third Division North'
third_div_north = df[df['Division'] == 'Third Division North']

# Extract the 'Goals' column — it's the 5th column (index 4) in the data
# The column names are: ['Club', 'Season', 'Division', 'Apps', 'Goals', 'Apps', 'Goals', 'Apps', 'Goals']
# So 'Goals' is at index 4
goals_values = third_div_north.iloc[:, 4]  # Goals column

# Calculate variance
variance = goals_values.var()

print(f"Final Answer: {variance:.2f}")
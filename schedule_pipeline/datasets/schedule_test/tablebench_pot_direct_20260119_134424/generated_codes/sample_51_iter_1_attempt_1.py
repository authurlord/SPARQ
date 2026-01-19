import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where Goal Difference is 16
filtered_teams = df[df['Goal Difference'] == 16]

# Extract the club names
teams = filtered_teams['Club'].tolist()

# Print the final answer
print(f"Final Answer: {', '.join(teams)}")
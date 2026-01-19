import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Goal Difference is exactly 16
teams_with_diff_16 = df[df['Goal Difference'].str.contains('16', na=False)]
# Extract club names
clubs = teams_with_diff_16['Club'].tolist()
print(f"Final Answer: {', '.join(clubs)}")
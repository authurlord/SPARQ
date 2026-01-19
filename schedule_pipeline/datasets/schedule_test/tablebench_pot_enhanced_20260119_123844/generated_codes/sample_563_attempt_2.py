import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the maximum number of major hurricanes
max_major_hurricanes_row = df.loc[df['number of major hurricanes'].idxmax()]
# Get the number of tropical storms for that year
tropical_storms = max_major_hurricanes_row['number of tropical storms']
print(f"Final Answer: {tropical_storms}")
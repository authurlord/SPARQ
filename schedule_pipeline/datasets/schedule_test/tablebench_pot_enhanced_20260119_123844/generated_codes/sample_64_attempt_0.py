import pandas as pd

df = pd.read_csv('table.csv')
# Filter for nations with 1 gold, 0 silver, and 0 bronze
filtered_nations = df[(df['gold'] == 1) & (df['silver'] == 0) & (df['bronze'] == 0)]
# Extract the nation names
nations = filtered_nations['nation'].tolist()
print(f"Final Answer: {', '.join(nations)}")
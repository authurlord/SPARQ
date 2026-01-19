import pandas as pd

df = pd.read_csv('table.csv')
# Count years with 3 or fewer hurricanes
count = df[df['number of hurricanes'] <= 3].shape[0]
print(f"Final Answer: {count}")
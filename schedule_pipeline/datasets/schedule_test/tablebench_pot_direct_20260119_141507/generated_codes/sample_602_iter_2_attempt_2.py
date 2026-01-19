import pandas as pd

df = pd.read_csv('table.csv')
# Find the total passengers at Glasgow International Airport
glasgow_passengers = df[df['airport'] == 'glasgow international']['total passengers'].values[0]
# Calculate 15% increase
increased_passengers = glasgow_passengers * 1.15
print(f"Final Answer: {int(increased_passengers)}")
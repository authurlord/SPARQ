import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'Gross' to numeric by removing $ and commas
df['Gross'] = df['Gross'].str.replace('$', '').str.replace(',', '').astype(float)

# Select top 10 movies by rank
top_10 = df.head(10)

# Create a pie chart of the gross earnings
plt.figure(figsize=(8, 8))
plt.pie(top_10['Gross'], labels=top_10['Title'], autopct='%1.1f%%', startangle=90)
plt.title('Box Office Earnings of Top 10 Ranked Movies')
plt.show()
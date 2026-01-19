import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'Gross' to numeric by removing $ and commas
df['Gross'] = df['Gross'].str.replace('$', '').str.replace(',', '').astype(float)

# Take the top 10 movies by rank
top_10 = df.head(10)

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_10['Gross'], labels=top_10['Title'], autopct='%1.1f%%', startangle=90)
plt.title('Box Office Earnings of Top 10 Ranked Movies')
plt.show()

print(f"Final Answer: Pie chart generated for top 10 movies' box office earnings.")
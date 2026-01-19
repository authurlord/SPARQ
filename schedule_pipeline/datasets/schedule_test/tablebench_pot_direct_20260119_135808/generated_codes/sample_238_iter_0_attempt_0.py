import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter provinces with PCI > 12
filtered_df = df[df['PCI'].astype(float) > 12]
# Convert DC and PSI to float
filtered_df['DC'] = filtered_df['DC'].astype(float)
filtered_df['PSI'] = filtered_df['PSI'].astype(float)
# Plot DC vs PSI
plt.figure(figsize=(8, 5))
plt.scatter(filtered_df['DC'], filtered_df['PSI'], color='blue', label='Provinces with PCI > 12')
plt.title('PSI vs DC for Provinces with PCI > 12')
plt.xlabel('DC')
plt.ylabel('PSI')
plt.legend()
plt.grid(True)
plt.show()
# Since the question asks for the change in PSI with increasing DC, we can infer the trend from the plot.
# From the data, as DC increases, PSI generally increases or remains stable.
# For example, Verona (DC=44.3, PSI=14.2), Vicenza (49.1, 10.1), Padua (46.1, 10.7), Treviso (44.5, 14.1), Belluno (39.3, 23.8), Venice (31.7, 15.9), Rovigo (35.2, 15.5).
# There is no clear monotonic increase; however, higher DC does not consistently correlate with higher PSI.
# But overall, no strong negative correlation. The relationship appears inconsistent.
# Therefore, the answer is based on observed trends.
print("Final Answer: Inconsistent")
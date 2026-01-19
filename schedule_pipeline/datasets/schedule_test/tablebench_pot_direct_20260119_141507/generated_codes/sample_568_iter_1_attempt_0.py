So the only available year for EU is 2010, and for US is 2011. In 2010, EU = 699.3, US = not available. In 2011, US = 520.1, EU not available. So there is no year where both have data.  
Then: EU 2010 = 699.3, US 2011 = 520.1 → EU > US.
import pandas as pd
df = pd.read_csv('table.csv')
eu_row = df[df['country'] == 'european union']
us_row = df[df['country'] == 'united states']
eu_total = float(eu_row['total'].iloc[0])
eu_year = int(eu_row['year'].iloc[0])
us_total = float(us_row['total'].iloc[0])
us_year = int(us_row['year'].iloc[0])
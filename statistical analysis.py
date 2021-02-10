import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'

df=pd.read_csv(path)

df.head(5)
df.index.name='test' #just a test to rename the index column
#to find correlation between different columns
df[['bore', 'stroke','compression-ratio' , 'horsepower']].corr()
scipy.stats.pearsonr(df['wheel-base'], df['price'])
#different plots to visualize the data

sns.regplot('bore', 'stroke', data=df)
sns.boxplot(x='body-style',y='price',data=df)

df_group_one = df[['drive-wheels','body-style','price','stroke']]

df_group_one=df_group_one.groupby('drive-wheels').mean()
df_group_one

df_group_one = df[['drive-wheels','body-style','price']]

df_group_two=df_group_one.groupby(['drive-wheels','body-style'],as_index=False).mean()
df_group_three=df_group_two.pivot(index="drive-wheels",columns='body-style')
df_group_three=df_group_three.fillna(0)
df_group_three


fig,ax=plt.subplots()
im=ax.pcolor(df_group_three,cmap='RdBu')


column_names=df_group_three.index
row_names=df_group_three.columns.levels[1]

ax.set_xticks(np.arange(df_group_three.shape[1])+0.5,minor=False)
ax.set_yticks(np.arange(df_group_three.shape[0]) + 0.5, minor=False)

ax.set_xticklabels(row_names)
ax.set_yticklabels(column_names)
plt.xticks(rotation=75)

plt.colorbar(im)



grouped_test=df[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test.head(2)

grouped_test.get_group('4wd')["price"]

scipy.stats.f_oneway(grouped_test.get_group('4wd')["price"],grouped_test.get_group('4wd')["price"],grouped_test.get_group('4wd')["price"])
scipy.stats.f_oneway(grouped_test.get_group('4wd')["price"],grouped_test.get_group('4wd')["price"])

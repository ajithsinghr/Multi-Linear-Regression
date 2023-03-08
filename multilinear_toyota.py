# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 21:38:47 2023

@author: ramav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("D:\\Assignments\\Multi linear\\ToyotaCorolla.csv",encoding="latin1")
df.head()
df.isnull().sum()
df.shape
df.info()

# extract only required which is mostly useful predicting y variable(price) from the given data

df.info()
df=pd.concat([df.iloc[:,2:4],df.iloc[:,6:7],df.iloc[:,8:9],df.iloc[:,12:14],
              df.iloc[:,15:18]],axis=1)
df

df=df.rename({'Age_08_04':'Age'},axis=1)
df

df[df.duplicated()]

df=df.drop_duplicates().reset_index(drop=True)
df

df.describe()

# Correlation Analysis
df.corr()

sns.set_style(style='darkgrid')
sns.pairplot(df)

# Model Building
import statsmodels.formula.api as smf
model=smf.ols('Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit()

# Model Testing
#  Coefficient 
model.params

#  tvalues and pvalues
model.tvalues , np.round(model.pvalues,5)

# Finding rsquared values
model.rsquared , model.rsquared_adj 

slr_c=smf.ols('Price~CC',data=df).fit()
slr_c.tvalues , slr_c.pvalues

slr_d=smf.ols('Price~Doors',data=df).fit()
slr_d.tvalues , slr_d.pvalues 

mlr_cd=smf.ols('Price~CC+Doors',data=df).fit()
mlr_cd.tvalues , mlr_cd.pvalues

# Model Validation Techniques
# Two Techniques: 1. Collinearity Check & 2. Residual Analysis

# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables

rsq_age=smf.ols('Age~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('cc~Age+KM+HP+Doors+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+cc+Doors+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('Quarterly_Tax~Age+KM+HP+cc+Doors+Gears+Weight',data=df).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax',data=df).fit().rsquared
vif_WT=1/(1-rsq_WT)

# Putting the values in Dataframe format
d1={'Variables':['Age','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'],
    'Vif':[vif_age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
Vif_df=pd.DataFrame(d1)
Vif_df

# 2) Residual Analysis
# Test for Normality of Residuals (Q-Q Plot) using residual model (model.resid)
import statsmodels.api as sm
sm.qqplot(model.resid,line='q') # 'q' - A line is fit through the quartiles # line = '45'- to draw the 45-degree diagonal line
plt.title("Normal Q-Q plot of residuals")
plt.show()

list(np.where(model.resid>6000))

list(np.where(model.resid<-6000))

# Test for Homoscedasticity or Heteroscedasticity (plotting model's standardized fitted values vs standardized residual values)

def standard_values(vals) : return (vals-vals.mean())/vals.std()

plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 

# Test for errors or Residuals Vs Regressors or independent 'x' variables or predictors 
# using Residual Regression Plots code graphics.plot_regress_exog(model,'x',fig)  

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Age',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'KM',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'HP',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'cc',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Doors',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Gears',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Quarterly_Tax',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Weight',fig=fig)
plt.show()


# Model Deletion Diagnostics (checking Outliers or Influencers)
# Two Techniques : 1. Cook's Distance & 2. Leverage value
# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
from statsmodels.graphics.regressionplots import influence_plot
(c,_)=model.get_influence().cooks_distance
c

# Plot the influencers using the stem plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(df)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()

# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)

# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
fig,ax=plt.subplots(figsize=(20,20))
fig=influence_plot(model,ax = ax)

# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=df.shape[1]
n=df.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff

# Improving the Model
# Creating a copy of data so that original dataset is not affected
df_new=df.copy()
df_new

# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
toyo_1=df_new.drop(df_new.index[[80]],axis=0).reset_index(drop=True)
toyo_1

# Model Deletion Diagnostics and Final Model
while np.max(c)>0.5 :
    model=smf.ols('Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyo_1).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    toyo_1=toyo_1.drop(toyo_1.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    toyo_1
else:
    final_model=smf.ols('Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyo_1).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)

if np.max(c)>0.5:
    model=smf.ols('Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyo_1).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    toyo_1=toyo_1.drop(toyo_1.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    toyo_1 
elif np.max(c)<0.5:
    final_model=smf.ols('Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyo_1).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)

final_model.rsquared

toyo_1

# Model Predictions
# say New data for prediction is
new_data=pd.DataFrame({'Age':12,"KM":40000,"HP":80,"cc":1300,"Doors":4,"Gears":5,"Quarterly_Tax":69,"Weight":1012},index=[0])
new_data

# Manual Prediction of Price
final_model.predict(new_data)

# Automatic Prediction of Price with 90.02% accurcy
pred_y=final_model.predict(toyo_1)
pred_y

'''
# knowing the outliers

for i in df:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3-Q1
    UW = Q3 + 1.5*IQR
    LW = Q1 - 1.5*IQR
    
    if df[(df[i]>UW) | (df[i]<LW)].any(axis=None):
        print(i, "yes")
    else:
        print(i, "no")
        
# BOX PLOT    
sns.boxplot(df["Price"],color="blue")    
sns.boxplot(df["Age"],color="black")    
sns.boxplot(df["Mfg_Year"],color="green")    
sns.boxplot(df["KM"],color="violet")    
sns.boxplot(df["HP"],color="red")    
sns.boxplot(df["cc"],color="yellow")    
sns.boxplot(df["Gears"],color="pink")    
sns.boxplot(df["Quarterly_Tax"],color="blue")

# Removing of outliers using for loop

outliers = ["Price","Age","KM","HP","cc","Gears","Quarterly_Tax"]
   
for i in df.loc[:,outliers]:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3-Q1
    UW = Q3 + 1.5*IQR
    LW = Q1 - 1.5*IQR
    df.loc[df[i]>UW,i] = UW
    df.loc[df[i]<LW,i] = LW

# Box plot after removing of outliers
sns.boxplot(df["Price"],color="blue")    
sns.boxplot(df["Age"],color="black")    
sns.boxplot(df["Mfg_Year"],color="green")    
sns.boxplot(df["KM"],color="violet")    
sns.boxplot(df["HP"],color="red")    
sns.boxplot(df["cc"],color="yellow")    
sns.boxplot(df["Gears"],color="pink")    
sns.boxplot(df["Quarterly_Tax"],color="blue")    
        
sns.set_style(style="darkgrid")
sns.pairplot(df)    

sns.histplot(df)       
plt.hist(df)    

# using pearson correlation
corr = df.corr()  
plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=True,cmap="magma")  # heat map
plt.show()

# with the following function i will select the highly correlated features
def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
        return(col_corr)
    
corr_features = correlation(df,0.85)
len(set(corr_features))

corr_features

# there is no much correlated independent varaibles so i will go for next step

# spliting into dependent(y) and independent(x) varaiables
df.head()
x = df.iloc[:,1:]
y = df["Price"]       

'''

        
        
        
        
        
        
#!/usr/bin/env python
# coding: utf-8

# In[557]:


##################################################################################
################################## Part I - Importing
##################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[558]:


df = pd.read_csv('HR_comma_sep.csv')


# In[559]:


df.head()


# In[560]:


########################################################
################# Part II - Duplicates
########################################################


# In[561]:


df[df.duplicated()].head()                         #### wasn't expecting this much honestly


# In[562]:


df = df.drop_duplicates()


# In[563]:


df[df.duplicated()]                  #### no duplicates


# In[564]:


df.info()


# In[565]:


df.left.value_counts()


# In[566]:


df.Department.unique()


# In[567]:


df.Department.nunique()


# In[446]:


###############################################################
############################### Part III - Missing Data
###############################################################


# In[568]:


fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='summer',ax=ax)

ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


#### seems like we are working with a very clean data here


# In[569]:


df.isnull().any()

#### making sure it got no empty or null data


# In[449]:


#########################################################
################# Part IV - EDA
#########################################################


# In[570]:


df.head()


# In[571]:


df.satisfaction_level.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('HR dataset')

plt.xlabel('Number of employees')

plt.ylabel('satisfaction level')


#### although this is too much data in one place but still we can see that majority of them are well satisfied


# In[572]:


pl = df.satisfaction_level.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

pl.set_xlim(0,800)

plt.title('HR dataset')

plt.xlabel('Number of employees')

plt.ylabel('satisfaction level')


#### from this subsection we see the satisfaction level is around 0.4 but if we take another slice of the data we will see different result


# In[573]:


pl = df.satisfaction_level.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

pl.set_xlim(2000,3000)

plt.title('HR dataset')

plt.xlabel('Number of employees')

plt.ylabel('satisfaction level')


#### now we see around 0.6 satisfaction level because we took difference slice of that data column, lets see the mean and std of them


# In[574]:


df.satisfaction_level.mean()          #### mean of satisfaction level


# In[575]:


df.satisfaction_level.std()           #### so the std is + or - 0.24 on either z score


# In[576]:


df.head()


# In[577]:


mean_df = df.satisfaction_level.mean()
std_df = df.satisfaction_level.std()


# In[578]:


#### now we will make a very nice comprehensive standard deviation graph for Age column
from scipy.stats import norm


x = np.linspace(mean_df - 4*std_df, mean_df + 4*std_df, 1000)
y = norm.pdf(x, mean_df, std_df)

#### plot
plt.figure(figsize=(12, 6))

#### normal distribution curve
plt.plot(x, y, label='Normal Distribution')


#### this is very basic one but as we feeling fancy today so we will do a very comprehensive one


# In[579]:


#### Comprehensive time

x = np.linspace(mean_df - 4*std_df, mean_df + 4*std_df, 1000)
y = norm.pdf(x, mean_df, std_df)

#### plot
plt.figure(figsize=(20, 7))

#### normal distribution curve
plt.plot(x, y, label='Normal Distribution')

#### areas under the curve
plt.fill_between(x, y, where=(x >= mean_df - std_df) & (x <= mean_df + std_df), color='green', alpha=0.2, label='68%')
plt.fill_between(x, y, where=(x >= mean_df - 2*std_df) & (x <= mean_df + 2*std_df), color='orange', alpha=0.2, label='95%')
plt.fill_between(x, y, where=(x >= mean_df - 3*std_df) & (x <= mean_df + 3*std_df), color='yellow', alpha=0.2, label='99.7%')

#### mean and standard deviations
plt.axvline(mean_df, color='black', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - std_df, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + std_df, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - 2*std_df, color='orange', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + 2*std_df, color='orange', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - 3*std_df, color='yellow', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + 3*std_df, color='yellow', linestyle='dashed', linewidth=1)

plt.text(mean_df, plt.gca().get_ylim()[1]*0.9, f'Mean: {mean_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + std_df, plt.gca().get_ylim()[1]*0.05, f'z=1    {mean_df + std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - std_df, plt.gca().get_ylim()[1]*0.05, f'z=-1   {mean_df - std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 2*std_df, plt.gca().get_ylim()[1]*0.05, f'z=2  {mean_df + 2*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 2*std_df, plt.gca().get_ylim()[1]*0.05, f'z=-2 {mean_df - 2*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 3*std_df, plt.gca().get_ylim()[1]*0.05, f'z=3  {mean_df + 3*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 3*std_df, plt.gca().get_ylim()[1]*0.05, f'z=-3 {mean_df - 3*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')


#### annotate the plot
plt.text(mean_df, max(y), 'Mean', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - std_df, max(y), '-1σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + std_df, max(y), '+1σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 2*std_df, max(y), '-2σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 2*std_df, max(y), '+2σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 3*std_df, max(y), '-3σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 3*std_df, max(y), '+3σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')

#### labels
plt.title('Satisfaction Level distribution inside the HR Dataset')
plt.xlabel('Satisfaction Level')
plt.ylabel('Probability Density')

plt.legend()


# In[580]:


#### lets say we wanna find out, the satisfaction level of employees with a confidence interval of 95%

standard_error = std_df/np.sqrt(df.shape[0])


# In[581]:


#### 95% confidence interval satisfaction level of employees in the HR dataset

from scipy import stats

stats.norm.interval(alpha=0.95,loc=mean_df,scale=standard_error)


# In[582]:


df.head()


# In[583]:


#### we wanna go one step further and see the confidence interval of 99%
#### 99% confidence interval 

stats.norm.interval(alpha=0.99,loc=mean_df,scale=standard_error)

#### this is quite interesting because even at the worst case scenario the satisfaction level doesn't go below 60% or 0.60


# In[584]:


pl = sns.FacetGrid(df,hue='number_project',aspect=4,height=4)

pl.map(sns.kdeplot,'satisfaction_level',fill=True)

pl.set(xlim=(0,df.satisfaction_level.max()))

pl.add_legend()

#### very interesting, the more projects you are involved in, the less satisfied you tend to be


# In[585]:


pl = sns.FacetGrid(df,hue='salary',aspect=4,height=4)

pl.map(sns.kdeplot,'average_montly_hours',fill=True)

pl.set(xlim=(0,df.average_montly_hours.max()))

pl.add_legend()


# In[586]:


df.head()


# In[587]:


#### lets see how things scale up when satisfaction level is compared to projects and who left the company

custom = {0:'green',
         1:'black'}

g = sns.jointplot(x=df.satisfaction_level,y=df.number_project,data=df,hue='left',palette=custom)

g.fig.set_size_inches(17,9)


#### quite revealing that anybody who was involved in 7 projects they all left the company
#### on satisfaction level graph we see a spike in people leaving the company when they are not satisfied which makes sense
#### also we see a spike on number of projects and it points that people who did around 2-3 projects also left


# In[588]:


custom = {0:'green',
         1:'black'}

g = sns.jointplot(x=df.satisfaction_level,y=df.average_montly_hours,data=df,hue='left',palette=custom)

g.fig.set_size_inches(17,9)


# In[589]:



custom = {'low':'grey',
          'medium':'pink',
          'high':'green'}

sns.catplot(x='left',y='satisfaction_level',data=df,kind='box',height=7,aspect=2,legend=True,hue='salary',palette=custom)


#### people who didn't leave the company most of them had higher satisfaction level
#### for people who left it didn't matter if they were paid low or medium, the main issue was satisfaction level
#### for people who left the company on higher salary,their satisfaction levels mean was higher then low and medium paid employees
#### also there are outliers in higher paid employees who left the company which we wouldn't take into account because they are outliers


# In[590]:


g = sns.jointplot(x='satisfaction_level',y='number_project',data=df,kind='reg',color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)


#### same it seems like people are more satisfied when the projects are less then 4


# In[591]:


#### lets make a Null hypothesis that satisfaction_level and number of projects are not correlated and they occur by chance
#### lets debunk this null hypothesis by introducing pearsonr

from scipy.stats import pearsonr


# In[592]:


co_eff, p_value = pearsonr(df.satisfaction_level,df.number_project)


# In[593]:


co_eff


# In[594]:


p_value              #### p-value < 0.05 significance level therefore we reject null hypothesis and accept alternative hypothesis


# In[596]:



sns.catplot(x='left',y='number_project',data=df,kind='box',height=7,aspect=2,legend=True,hue='Department',palette='Set3')


#### its pretty clear that people who left the company were involved in more then 4 projects


# In[597]:


custom = {0:'green',
         1:'black'}

sns.catplot(x='number_project',y='satisfaction_level',data=df,kind='strip',height=7,aspect=2,legend=True,hue='left',jitter=True,palette=custom)


#### see here its getting confusing, we know for a fact that satisfaction level maintains well when the number of projects are 3-5
#### but we see people leaving the company when the projects drop below 3 and increases above 5, so in short keep projects 3-5 but 3 being the most safest spot


# In[598]:


corr = df.corr()

fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


#### as our best interest is in the left column feature so from that perspective it seems left is the most correlated to time spent in the company
#### so lets explore that aspect now


# In[599]:


custom = {0:'green',
          1:'red'}

pl = sns.FacetGrid(df,hue='left',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'time_spend_company',fill=True)

pl.set(xlim=(0,df.time_spend_company.max()))

pl.add_legend()


#### seems like employees who spent more years into the company don't leave the company, most of the spike in left is between 2-3 years
#### once they hit that milestone the probability they will leave the company goes down significantly


# In[600]:


custom = {0:'green',
          1:'red'}

sns.catplot(x='time_spend_company',data=df,hue='left',palette=custom,kind='count',height=7,aspect=1.7)


#### seems like our predictions were right


# In[601]:


custom = {'low':'grey',
          'medium':'pink',
          'high':'green'}

sns.catplot(x='Department',data=df,hue='salary',kind='count',height=7,aspect=1.7,palette=custom)


#### seems like management department is getting paid the most ratio wise


# In[602]:


custom = {0:'purple',
          1:'green'}

pl = sns.catplot(y='left',x='time_spend_company',data=df,kind='point',height=10,aspect=1.5,hue='promotion_last_5years',palette=custom)

plt.xticks([1,2,3,4,5,6,7,8,9,10])


#### seems like people who got the promotion didn't leave the company


# In[603]:


custom = {0:'black',
          1:'green'}


sns.lmplot(x='time_spend_company',y='left',data=df,hue='promotion_last_5years',palette=custom,x_bins=[range(1,10)],height=6,aspect=2)

#### we further prove how important promotion is to making employees not leave the company linear model plotting


# In[604]:


#### lets do some feature engineering and see if Department is somehow related to people leaving the company

custom = {0:'green',
          1:'purple'}

sns.catplot(x='Department',data=df,kind='count',height=10,aspect=1.5,hue='left',palette=custom)


#### seems like sales department has the most leaving and also most not leaving the company


# In[605]:


df.Department.value_counts()           #### we have more sales employees then any other department so it doesn't suprise me


# In[606]:


df.Department.unique()


# In[607]:


map_dept = {'sales':0,
            'accounting':1,
            'hr':2,
            'technical':3,
            'support':4,
            'management':5,
            'IT':6,
            'product_mng':7,
            'marketing':8,
            'RandD':9}

df['Dept'] = df['Department'].map(map_dept)


# In[608]:


map_income = {'low':0,
              'medium':1,
              'high':2}

df['Sal'] = df.salary.map(map_income)


# In[609]:


df.head()


# In[610]:


df.info()


# In[611]:


df.Sal.value_counts()


# In[612]:


df.Dept.value_counts()


# In[613]:


corr = df.corr()


# In[614]:


fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


#### it seems like salary isn't part of why people leave the company as they are negatively correlated, pretty revealing
#### left feature column is highly correlated to number of years an employee is in company


# In[615]:


#### lets explore salary column now and its impact if any, according to correlation its negatively correlated which means it doesn't have any or much effect

custom = {'low':'red',
          'medium':'orange',
          'high':'green'}

sns.lmplot(x='time_spend_company',y='left',data=df,hue='salary',x_bins=[range(1,10)],height=6,aspect=2,palette=custom)


#### you see when we do just the correlation and then see left feature and salary feature then we don't see much but once we explore with regards to time spent then we start seeing the real difference
#### in short people who were paid low were more likely to leave the company


# In[495]:


#### here we will go systematically following the heatmap and correlation in regards to feature column and target columns
#### Part I - satisfaction level


# In[616]:


#### satisfaction level is some correlation with last_evaluation and work accident

custom = {0:'green',
          1:'red'}

sns.lmplot(x='satisfaction_level',y='left',data=df,hue='Work_accident',palette=custom,height=7,aspect=2,x_bins=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

#### seems like work accident didn't have much of an influence in people leaving the company


# In[617]:


sns.lmplot(x='satisfaction_level',y='left',data=df,hue='Department',palette='Set1',height=7,aspect=2,x_bins=[0.1,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.43,0.45,0.47,0.49,0.5,0.55,0.59,0.6,0.65,0.69,0.7,0.74,0.78,0.8,0.84,0.89,0.9,0.95,0.99,1])


#### quite revealing


# In[618]:


##############################################
############# Part V - PCA
##############################################


# In[619]:


X = df.drop(columns=['left','Department','salary'])


# In[620]:


X.head()


# In[621]:


y = df['left']

y.head()


# In[622]:


y.value_counts()


# In[623]:


#### PCA time

from sklearn.preprocessing import StandardScaler


# In[624]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[625]:


from sklearn.decomposition import PCA


# In[626]:


pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])
final_df = pd.concat([principal_df, y], axis=1)


# In[627]:


final_df.head()                    #### the beauty of PCA, love it


# In[628]:


final_df.isnull().any()


# In[629]:


final_df.principal_component_1.isnull().sum()


# In[630]:


final_df = final_df.dropna()


# In[631]:


final_df.isnull().any()


# In[634]:


colors = {0: 'black', 1: 'red'}

plt.figure(figsize=(15, 6))

for i in final_df['left'].unique():
    subset = final_df[final_df['left'] == i]
    plt.scatter(subset['principal_component_1'], subset['principal_component_2'], 
                color=colors[i], label=f'left = {i}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of HR Dataset')
plt.legend()
plt.grid(True)


#### see how you can make cluster or classfication with perfection with this method


# In[635]:


pca.n_features_                 #### number of feature columns that the PCA contains or was used for


# In[636]:


pca.components_


# In[637]:


X.columns


# In[638]:


df_comp = pd.DataFrame(pca.components_,columns=['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Dept','Sal'])


# In[639]:


df_comp.head()


# In[641]:


fig, ax = plt.subplots(figsize=(20,8))                     #### you see we have not given PCA the target column but even without that it was able to predict the correlation and make the cluster
                                                           #### its debatable what does 0 and 1 means here so to clarify we will make the model from this but this is the best method to see the correlation in my opinion
sns.heatmap(df_comp,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


# In[642]:


from sklearn.model_selection import train_test_split


# In[643]:


X = final_df.drop(columns='left')


# In[644]:


X.head()


# In[645]:


y = final_df.left


# In[646]:


y.head()


# In[647]:


y.value_counts()


# In[648]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[649]:


from sklearn.linear_model import LogisticRegression         #### for classification


# In[650]:


model = LogisticRegression()


# In[651]:


model.fit(X_train,y_train)


# In[652]:


y_predict = model.predict(X_test)


# In[653]:


from sklearn import metrics


# In[654]:


metrics.accuracy_score(y_test,y_predict)                 #### not bad for our PCA model, PCA is not used to model or predict
                                                         #### also this model is very very very basic, I can't emphasize that enough
                                                         #### so considering all those aspects the accuracy is pretty good


# In[655]:


print(metrics.classification_report(y_test,y_predict))        #### look at the recall and precision in predicting people who left company
                                                              #### it may be due to the fact we didn't normalize our target column and right now its sitting at float which shouldn't be the case for target columns
                                                              #### further down we will make it better using pipelines and converting into category with onehotencoder


# In[656]:


from sklearn.ensemble import RandomForestClassifier


# In[657]:


model = RandomForestClassifier()


# In[658]:


model.fit(X_train,y_train)


# In[659]:


y_predict = model.predict(X_test)


# In[660]:


metrics.accuracy_score(y_test,y_predict)


# In[661]:


print(metrics.classification_report(y_test,y_predict))             #### this is much better model, Randomforest will always be better then Logistic classifier


# In[662]:


metrics.confusion_matrix(y_test,y_predict)


# In[664]:


cm = metrics.confusion_matrix(y_test,y_predict)

labels = ['Not left','Left']

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

fig, ax = plt.subplots(figsize=(15,8))

disp.plot(ax=ax)

#### now we will go back to making the model how its usually done, usually PCA is not used to predict or make models on to it, but you can if you want to
#### but to challenge ourself we will go for the conventional way


# In[382]:


#########################################################################
######################### Part VI - Model - Classification
#########################################################################


# In[665]:


df.head()                 #### we will do the pipeline and to challenge ourself we will keep category columns as they are instead of choosing numerical ones


# In[666]:


X = df.drop(columns=['left','Dept','Sal'])


# In[667]:


X.head()                 #### feature columns


# In[668]:


y = df.left


# In[669]:


y.head()                      #### target


# In[670]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[671]:


X.columns


# In[672]:


preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), ['Department', 'salary']),
                                               ('num', StandardScaler(),['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years'])
                                              ]
                                )


#### this is my preferred method, I honestly don't like dummy variables, it makes the whole data frame so complicated and then you also have issues with collinearity
#### coming from R and its amazing factor() function I had to find something similar in python and so far this the best and closest to factor()
#### if you know better method then this please chip in


# In[673]:


from sklearn.pipeline import Pipeline


# In[674]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[675]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[676]:


model.fit(X_train,y_train)


# In[677]:


y_predict = model.predict(X_test)


# In[678]:


metrics.accuracy_score(y_test,y_predict)                     #### ok model for a very basic without any tunings involved


# In[679]:


print(metrics.classification_report(y_test,y_predict))


# In[680]:


############################################################
###################  Random forest
############################################################

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])


# In[681]:


model.fit(X_train,y_train)


# In[682]:


y_predict = model.predict(X_test)


# In[683]:


metrics.accuracy_score(y_test,y_predict)                      #### didnt expect this honestly, without tuning or anything we are getting almost perfect model with randomforest


# In[684]:


print(metrics.classification_report(y_test,y_predict))


# In[685]:


metrics.confusion_matrix(y_test,y_predict)


# In[687]:


cm = metrics.confusion_matrix(y_test,y_predict)

labels = ['Not left','Left']

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

fig, ax = plt.subplots(figsize=(15,8))

disp.plot(ax=ax)

#### this is a good model


# In[688]:


#### lets see if with some tuning or gridsearch we can achieve even better results

from sklearn.model_selection import GridSearchCV


# In[689]:


param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

model_grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
model_grid.fit(X_train, y_train)


# In[690]:


best_model = model_grid.best_estimator_


# In[691]:


y_predict = best_model.predict(X_test)


# In[692]:


metrics.accuracy_score(y_test,y_predict)                  #### went from 0.981 to 0.980


# In[693]:


print(metrics.classification_report(y_test,y_predict))


# In[694]:


############################################
############## KNN
############################################


# In[695]:


from sklearn.neighbors import KNeighborsClassifier


# In[696]:


get_ipython().run_cell_magic('time', '', "\nk_range = range(1,100)\n\naccuracy = []\n\nfor i in k_range:\n    \n    model = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('classifier', KNeighborsClassifier(n_neighbors=i))\n    ]) \n    \n    model.fit(X_train,y_train)\n    \n    y_predict = model.predict(X_test)\n    \n    accuracy.append(metrics.accuracy_score(y_test,y_predict))")


# In[697]:


plt.figure(figsize=(15,7))

plt.plot(k_range,accuracy,color='red', marker='o', linestyle='dashed',linewidth=2, markersize=10,markerfacecolor='black')

plt.xlabel('K Values')

plt.ylabel('Accuracy')

#### interesting as k value increases the accuracy decreases, wasn't expecting that honestly


# In[698]:


#### we know the best accuracy is between 30 and 50 from the plot


k_range = range(1,10)

accuracy = []

for i in k_range:
    
    model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=i))
    ]) 
    
    model.fit(X_train,y_train)
    
    y_predict = model.predict(X_test)
    
    accuracy.append(metrics.accuracy_score(y_test,y_predict))


# In[699]:


plt.figure(figsize=(15,7))

plt.plot(k_range,accuracy,color='red', marker='o', linestyle='dashed',linewidth=2, markersize=10,markerfacecolor='black')

plt.xlabel('K Values')

plt.ylabel('Accuracy')

#### seems k=2 is the best one here but still Gridsearch with randomforest yields the best model with accuracy of 0.98


# In[417]:



############################################################
############### Advanced Models
############################################################


# In[418]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[419]:


import xgboost as xgb


# In[420]:


clf_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

param_grid_xgb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.7, 0.8, 0.9],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9]
}


# In[421]:


from sklearn.model_selection import RandomizedSearchCV


# In[422]:


get_ipython().run_cell_magic('time', '', "\nrandom_search_xgb = RandomizedSearchCV(clf_xgb, param_grid_xgb, n_iter=50, cv=5, scoring='accuracy', random_state=42)\nrandom_search_xgb.fit(X_train, y_train)")


# In[423]:


best_model = random_search_xgb.best_estimator_


# In[424]:


y_predict = best_model.predict(X_test)


# In[425]:


metrics.accuracy_score(y_test,y_predict)


# In[426]:


print(metrics.classification_report(y_test,y_predict))


# In[256]:


#### last one with stacking several models together


# In[427]:


from sklearn.ensemble import StackingClassifier


# In[428]:


base_models = [
    ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

meta_model = LogisticRegression()

stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)


# In[429]:


get_ipython().run_cell_magic('time', '', "\nmodel = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('classifier', stacking_clf)\n])\n\nmodel.fit(X_train, y_train)")


# In[430]:


y_predict = model.predict(X_test)


# In[432]:


metrics.accuracy_score(y_test, y_predict)           #### seems like this is the best we can get from this data set
                                                    #### went from 0.9813 to 0.9819


# In[433]:


print(metrics.classification_report(y_test,y_predict))


# In[434]:


#### it seems like now our model is not improving much, well there was not much to improve to begin with as our model had almost perfect accuracy from the start
#### but we produced the best one with stacking classifier model, giving us the accuracy of 0.9819


# In[ ]:





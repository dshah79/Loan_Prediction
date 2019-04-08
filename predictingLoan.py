import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

'''
Pandas output Setting 

'''
pd.set_option('display.max_columns',20)

should_print = False   ### Control Variable for printing

should_plot  = False   ### Control Variable for plot the graph

dataFile = "train_LoanPredDataset.csv"

df = pd.read_csv(dataFile)

should_print and print (df.dtypes)

should_print and print (df.describe())   ## quantitative Analysis

'''
LoanAmount -> 22 Value Missing
Loan_Amount_Term -> 14 Value Missing
Credit_history ->  50 Value Missing

mean > 50%

50% Salary less then Mean 
'''

'''
value_counts -> Freq Distubution 
'''

#print (df['Property_Area'].value_counts())
'''

'''

#print (df['Education'].describe())
'''
count          614
unique           2
top       Graduate
freq           480
Name: Education, dtype: object

Mariority of Graduate apply for loans 
'''

fig = plt.figure(figsize=(8,4))

should_plot and df['ApplicantIncome'].hist(bins=50)

should_plot and df.boxplot(column='ApplicantIncome')

should_plot and df.boxplot(column='ApplicantIncome', by='Education')


credit_History_feq = df['Credit_History'].value_counts(ascending=True)
'''
0.0     89
1.0    475
Name: Credit_History
'''

pivot_Table = df.pivot_table(values='Loan_Status',
                             index=['Credit_History'],
                             aggfunc=lambda x: x.map({'Y':1, 'N':0}).mean())

'''
print (pivot_Table)
                Loan_Status
Credit_History             
0.0                0.078652
1.0                0.795789
'''


fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(121)

ax1.set_xlabel("Credit History")

ax1.set_ylabel("Count of Application")

ax1.set_title("Applicants by Credit_History")

credit_History_feq.plot(kind='bar')

ax2 = fig.add_subplot(122)

pivot_Table.plot(kind='bar')

ax2.set_xlabel("Credit History")

ax2.set_ylabel("Probability of getting loan")

ax2.set_title("Probability of getting loan by credit History")

married_feq_count = df['Married'].value_counts()

pivot_Table_Married = df.pivot_table(values='Loan_Status',
                                     index=['Married'],
                                     aggfunc=lambda x: x.map({'Y':1, 'N':0}).mean())

fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(121)

ax1.set_xlabel("Credit History")

ax1.set_ylabel("Count of Application")

ax1.set_title("Applicants by Credit_History")

credit_History_feq.plot(kind='bar')

ax2 = fig.add_subplot(122)

pivot_Table.plot(kind='bar')

ax2.set_xlabel("Credit History")

ax2.set_ylabel("Probability of getting loan")

ax2.set_title("Probability of getting loan by Married")

crosstab = pd.crosstab(df['Credit_History'], df['Loan_Status'])

crosstab.plot(kind='bar', stacked=True, color=['red', 'blue'], grid=False)

'''

'''
crosstab = pd.crosstab([df['Credit_History'],df['Married']], df['Loan_Status'])

crosstab.plot(kind = 'bar', stacked=True, color=['red', 'Yellow'], grid=False)


#print (df.apply(lambda x: sum(x.isnull())))
##Filling Up
#df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)


'''
sns.set_style("whitegrid")

ax = sns.boxplot(
                 x='Self_Employed', 
                 y="LoanAmount", 
                 hue="Education",
                 data=df,palette="Set3"
                )
                
Trying to Find if there is an realtionship between Education and Self_employeed
c
rosstab = pd.crosstab(df['Education'], df['Self_Employed'])

crosstab.plot(kind='bar', stacked=True, color=['red', 'blue'], grid=False)

print (df.pivot_table(values='Self_Employed', index=['Education'], aggfunc=lambda x : x.map({'Yes':1, 'No': 0}).mean()))

              Self_Employed
Education                  
Graduate           0.143172
Not Graduate       0.132812

print (df['Self_Employed'].value_counts())


dtype: int64
No     500
Yes     82

it safe to put No -> 85% are not self Employee 
'''

df['Self_Employed'].fillna("No",inplace=True)



loanAmountMissingValue = ( df.pivot_table(values='LoanAmount',
                index='Self_Employed', columns='Education', aggfunc=np.median))

'''
Self_Employed     No    Yes
Education                  
Graduate       130.0  157.5
Not Graduate   113.0  130.0
'''

def fillingValue(dataFrame):
    return loanAmountMissingValue.loc[dataFrame['Self_Employed'],dataFrame['Education']]


df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fillingValue,axis=1), inplace=True)

#df['Gender'].value_counts()
df['Gender'].fillna('Male', inplace=True)

#df['Married'].value_counts()
df['Married'].fillna('Yes', inplace=True)

#print (df['Loan_Amount_Term'].value_counts())
df['Loan_Amount_Term'].fillna(360.0, inplace=True)


df['Dependents'].fillna(0, inplace=True)

df['Credit_History'].fillna(1, inplace=True)

'''
Since the exterme values are practically possible 

So instead of treating them as outliers let's try Log transformation 
'''

fig = plt.figure(figsize=(8,4))

df['LoanAmount_log'] = np.log(df['LoanAmount'])

df['Total_income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

df['Total_income'] = np.log(df['Total_income'])

df['Total_income'].hist(bins=50)


should_plot and plt.show()







var_mod = ['Gender', 'Married' , 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
le = LabelEncoder()

for i in var_mod:
    df[i] = le.fit_transform(df[i])

#print (df[i].head())
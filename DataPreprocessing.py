import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from sklearn.preprocessing import StandardScaler

dir = './UTKFace/'

df = pd.DataFrame(columns=['filepath'], data = os.listdir(dir))
ages = []
genders = []

for i,val in enumerate(df['filepath']):
    txt = val.split("_")
    df['filepath'][i] = dir + val
    age = int(txt[0])
    # if (age<=10):
    #     ages.append(-1)
    # elif(age>10 and age<=15):
    #     ages.append(0)
    # elif(age>15 and age<=20):
    #     ages.append(1)
    # elif(age>20 and age<=25):
    #     ages.append(2)
    # elif(age>25 and age<=30):
    #     ages.append(3)    
    # elif(age>30 and age<=35):
    #     ages.append(4)    
    # elif(age>35 and age<=40):
    #     ages.append(5)    
    # elif(age>40 and age<=45):
    #     ages.append(6)
    # elif(age>45 and age<=50):
    #     ages.append(7)
    # elif(age>50 and age<=55):
    #     ages.append(8)
    # elif(age>55 and age<=60):
    #     ages.append(9)
    # elif(age>60 and age<=70):
    #     ages.append(10)
    # elif(age>70):
    #     ages.append(11)
    ages.append(int(txt[0]))
    genders.append(int(txt[1]))

# ss = StandardScaler()

df['age'] = ages
df['gender'] = genders

print(df)
df = df.drop(df[(df['age']>65)].index)
df = df.drop(df[(df['age']<15)].index)
# df = df.drop(df[(df['age']==0)].index)
# df = df.drop(df[(df['age']==11)].index)
print(len(df['age'].unique()))
# df.drop(indices,inplace=True)

pd.DataFrame.to_csv(df,'UTKFace2.csv', index=False)
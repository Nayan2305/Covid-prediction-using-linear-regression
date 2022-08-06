from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import style
import operator
from sklearn.model_selection import train_test_split
cases_confirmed=pd.read_csv("time_series_covid-19_confirmed.csv")
deaths_confirmed=pd.read_csv("time_series_covid-19_deaths.csv")
recovered_confirmed=pd.read_csv("time_series_covid-19_recovered.csv")

cases = cases_confirmed.iloc[:,4:]
deaths = deaths_confirmed.iloc[:,4:]
recovered = recovered_confirmed.iloc[:,4:]

dates = cases.keys()
total_cases = []
total_deaths = [] 
total_recovered = [] 

for i in dates:
    case_sum=cases[i].sum()
    total_cases.append(case_sum)
    death_sum=deaths[i].sum()
    total_deaths.append(death_sum)
    recovered_sum=recovered[i].sum() 
    total_recovered.append(recovered_sum)
  
days_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
total_cases = np.array(total_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)

forcast_days = 10
future_forecast = np.array([i for i in range(len(dates)+forcast_days)]).reshape(-1, 1)
adjusted_dates = future_forecast[:-10]



unique_countries =  list(cases_confirmed['Country/Region'].unique())
# For visualization with the latest data of 15th of march

latest_confirmed = cases_confirmed[dates[-1]]
latest_deaths = deaths_confirmed[dates[-1]]
latest_recoveries =recovered_confirmed[dates[-1]]

# The next line of code will basically calculate the total number of confirmed cases by each country

country_confirmed_cases = []
no_cases = []
for i in unique_countries:
    cases = latest_confirmed[cases_confirmed['Country/Region']==i].sum()
    if cases > 0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
        
for i in no_cases:
    unique_countries.remove(i)
    
unique_countries = [k for k, v in sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i] = latest_confirmed[cases_confirmed['Country/Region']==unique_countries[i]].sum()
# number of cases per country/region

# Plot a bar graph to see the total confirmed cases across different countries

plt.figure(figsize=(10, 100))
plt.barh(unique_countries, country_confirmed_cases)
plt.title('Number of Covid-19 Confirmed Cases in Countries',size=30)
plt.xlabel('Number of Covid19 Confirmed Cases')
plt.style.use('dark_background')
plt.show()


# Plot a bar graph to see the total confirmed cases between mainland china and outside mainland china 

china_confirmed = latest_confirmed[cases_confirmed['Country/Region']=='China'].sum()
outside_mainland_china_confirmed = np.sum(country_confirmed_cases) - china_confirmed

plt.barh('Mainland China', china_confirmed)
plt.barh('Outside Mainland China', outside_mainland_china_confirmed)
plt.title('Number of Confirmed Coronavirus Cases')
plt.style.use('dark_background')
plt.show()

# Only show 10 countries with the most confirmed cases, the rest are grouped into the category named others

visual_unique_countries = [] 
visual_confirmed_cases = []
others = np.sum(country_confirmed_cases[10:])
for i in range(len(country_confirmed_cases[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(country_confirmed_cases[i])

visual_unique_countries.append('Others')
visual_confirmed_cases.append(others)

# Create a pie chart to see the total confirmed cases in 10 different countries

plt.title('Covid-19 Confirmed Cases per Country')
plt.pie(visual_confirmed_cases)
plt.legend(visual_unique_countries, loc='best')
plt.style.use('dark_background')
plt.show()



#predicting cases
X_train_confirmed, X_test_confirmed,y_train_confirmed, y_test_confirmed=train_test_split(days_1_22,total_cases)
linear_cases = LinearRegression()
linear_cases.fit(X_train_confirmed, y_train_confirmed)
linear_case_pred = linear_cases.predict(future_forecast)
plt.plot(adjusted_dates,total_cases,color="cyan",linestyle='dashed')
plt.plot(future_forecast,linear_case_pred,color="red",linestyle=':')
plt.title("Cases over time",color="orange",size=30)
plt.xlabel("Days since 1/22/2020",color='orange')
plt.ylabel("Number of cases",color='orange')
plt.style.use('dark_background')
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()

#predicting deaths
X_train_confirmed, X_test_confirmed,y_train_confirmed, y_test_confirmed=train_test_split(days_1_22,total_deaths,test_size=0.15,shuffle=False)
linear_deaths = LinearRegression()
linear_deaths.fit(X_train_confirmed, y_train_confirmed)
linear_death_pred = linear_deaths.predict(future_forecast)
plt.plot(adjusted_dates,total_deaths,color="cyan",linestyle='dashed')
plt.plot(future_forecast,linear_death_pred,color="red",linestyle=':')
plt.title("Deaths over time",color="orange",size=30)
plt.xlabel("Days since 1/22/2020",color='orange')
plt.ylabel("Number of deaths",color='orange')
plt.xticks(size=15)
plt.yticks(size=15)
plt.style.use('dark_background')
plt.show()

# #predicting recovries
X_train_confirmed, X_test_confirmed,y_train_confirmed, y_test_confirmed=train_test_split(days_1_22,total_recovered,test_size=0.15,shuffle=False)
linear_recovered = LinearRegression()
linear_recovered.fit(X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_recovered.predict(X_test_confirmed)
linear_recovered_pred = linear_recovered.predict(future_forecast)
plt.plot(adjusted_dates,total_recovered,color="cyan",linestyle='dashed')
plt.plot(future_forecast,linear_recovered_pred,color="red",linestyle=':')
plt.title("Recoveries over time",color="orange",size=30)
plt.xlabel("Days since 1/22/2020",color='orange')
plt.ylabel("Number of recoveries",color='orange')
plt.xticks(size=15)
plt.yticks(size=15)
plt.style.use('dark_background')
plt.show()
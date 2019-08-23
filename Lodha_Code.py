#!/usr/bin/env python
# coding: utf-8

# The purpose of this script is the following: 
# 1. To load data from some online or flat file sources 
# 2. To organize the data so that there is a relationship with zip code 
# 3. To calculate various statistical metrics 
# 4. To create vizualizations such as plots and heat maps using these metrics.
# 
# This script will output various useful data sets that can be loaded as a csv. This script will also generate different visualizations to showcase what is happening in the data. 
# 
# 

# In[252]:


#### Libraries to Import

import pandas as pd
import numpy as np
import json
from pygeocoder import Geocoder
import os
import folium
import matplotlib.pyplot as plt


# In[253]:


#### Global Variables/ Hard Coded Values (Lookups)

geocoder=Geocoder('AIzaSyAACJSmDkcD9Avba8H_HDBQ21L2QGl0kNQ')
nyc_counties=['New York','Queens','Richmond','Kings','Bronx']
financials=['zipcode','agi_stub','N1','mars1','MARS2','MARS4','A00100']   ## Which Income Return Number to use?
gross_income_dict=dict([(1,1),(2,25),(3,50),(4,75),(5,100),(6,200)])

##### Custom Functions 

#Use Lat and Long to get zipcode 
def get_zip(row):
    if (pd.isna(row['Zip'])):
        result=geocoder.reverse_geocode(row['lat'],row['lon'])
        return result.postal_code
    else: 
        return row['Zip']

#Calculates the mode using the most returns for a given AGI category
def find_mode_agi(row):
    if row['N1']==row[('N1','max')]:
        return row['individual_agi']
    else: 
        return None

# Calculates correlation between the given metrics of and the number of playgrounds 
# in a given zip code. It calculates correlation in the whole city, all counties seperately, and the whole city not including manhattan
def correl_table(table):
    correlation_main=table.corr(method='pearson')
    correlation_table=correlation_main[['playground_count']].rename(columns={'playground_count':'All_NYC'})
    
    # Excluding manhattan
    table_no_manhattan=table[table[('County Name','')]!='New York']
    correlation_no_newyork=table_no_manhattan.corr(method='pearson')
    correlation_no_newyork=correlation_no_newyork[['playground_count']].rename(columns={'playground_count':'No_Manhattan'})
    correlation_table=correlation_table.merge(correlation_no_newyork,left_index=True,right_index=True,how='left')
    
    list_of_counties=table[('County Name','')].unique()
    
    # Correlation by county
    for county in list_of_counties:
        tmp_df=table[table[('County Name','')]==county]
        correlation_county=tmp_df.corr(method='pearson')
        correlation_column=correlation_county[['playground_count']].rename(columns={'playground_count':county})
        correlation_table=correlation_table.merge(correlation_column,left_index=True,right_index=True,how='left')
    
    correlation_table['standard deviation']=correlation_table.std(axis=1)
    
    return correlation_table

# This generates a heat map across NYC counties. It uses a geojson which contains 
# the borders of the NYC counties. It then calculates the quantiles at every .2 interval of the dataset 
# that needs to be visualized. It then generates the colors on top of the county borders. 
# The colors are divided based on the quantiles. 
def generate_map(table,key_column,value_column,legend_name):

    with open('Original Data/nyc_zip_code.geojson','r') as jsonFile:    #GEOJSON file in Original Data folder
        data=json.load(jsonFile)

    bins=list(table[value_column].quantile([0,.2,.4,.6,.8,.9,1]))    
    # print (bins)

    nyc_map=folium.Map(location=[40.7128,-74.0060],zoom_start=10)
    nyc_map.choropleth(
        geo_data=r'Original Data/nyc_zip_code.geojson',
        fill_opacity=0.7,
        line_opacity=0.2,
        data=table,
        key_on='feature.properties.postalCode',
        columns=[key_column,value_column],
        fill_color='YlOrRd',
        legend_name=legend_name,
        bins=bins
    )

    return nyc_map

#Creates a new id for playgrounds 
def generate_id(row):
    if (pd.isna(row['Playground_ID'])==False):
        row['id_type']='Park'
        return row['Playground_ID']
    elif (pd.isna(row['School_ID'])==False):
        row['id_type']='School'
        return ("School:"+row['School_ID'])
    else:
        row['id_type']='Unknown'
        return ("Park:"+row['Prop_ID'])


#  This next section of code collects data. Most sources are available online and the code automatically picks up from the website. However, copies of each of these datasets is available in the Original Data folder in the same directory as this code. The Geojson, however, is not collected in this section 

# In[254]:


# Collect Data

parks_raw=pd.read_json('https://www.nycgovparks.org/bigapps/DPR_Parks_001.json')
playgrounds_raw=pd.read_json('https://www.nycgovparks.org/bigapps/DPR_Playgrounds_001.json')
income_tax_raw=pd.read_csv('https://www.irs.gov/pub/irs-soi/16zpallagi.csv')
ny_zips_raw=pd.read_csv('https://data.ny.gov/api/views/juva-r6g2/rows.csv?accessType=DOWNLOAD')
# income_tax_raw=pd.read_csv('Original Data/16zpallagi.csv')    #Hard File


# This section of code creates a dataset which links zipcode to a clean playground id. For every playground id there is also a value called 'zip_code_count' This number states how many zip codes are associated with that give playground id 

# In[255]:


#Parks Data Cleanup = Creates a list of playgrounds into one list

parks=parks_raw[['Prop_ID','Zip']]

playgrounds_all=playgrounds_raw.drop_duplicates(['Prop_ID','Playground_ID','School_ID'])
playgrounds=playgrounds_all.merge(parks,on='Prop_ID',how='left')

###
# --> Additional playground clean up code goes here (extracting zip code from address)
####

# Find how many playgrounds wont result in a zipcode
number_of_nulls=playgrounds[playgrounds[['lat','lon','Zip']].isna().all(1)]['Playground_ID'].count()
percent_nulls="{0:.0%}".format(number_of_nulls/total_playgrounds)
print ("The Zip code for "+str(number_of_nulls)+" ("+percent_nulls+") could not be determined and is being dropped")

# Getting all the zips for every playground either through parks data or google api 
playgrounds=playgrounds.dropna(subset=['lat','lon','Zip'],how='all')           #Removing all addresses that wont result in zip -> Throws an error to Google API 
playgrounds['zip_clean']=playgrounds.apply(lambda row: get_zip(row),axis=1)    #Can this be parallelized?   "Swifter" Library
playgrounds['id_clean']=playgrounds.apply(lambda row: generate_id(row), axis=1)

playgrounds['zip_code_count']=playgrounds['zip_clean'].str.count(',')+1        #Calculates number of zips per playground

# Splitting all the zips which are a csv into multiple rows
playgrounds_by_zip=(playgrounds.set_index(['id_clean','zip_code_count'])).zip_clean.str.split(',',expand=True).stack().reset_index()                .rename(columns={0:'Zip'}).drop('level_2',axis=1)             # Converts the list of zip codes into individual lines
playgrounds_by_zip['Zip']=playgrounds_by_zip['Zip'].str.strip()
playgrounds_by_zip=playgrounds_by_zip.drop_duplicates()                       # Some of the parks have the same zip code listed multiple times. This removes all duplicates 

# This creates the first dataset which links every playground to its respective zip codes
playgrounds_by_zip.to_csv('playgrounds_by_zip.csv')


print ('Done')


# This section of code calculates some basic statistics on the playground data

# In[61]:


### Playgrounds Statistics


zip_code_distribution=playgrounds[['id_clean','zip_code_count']].groupby('zip_code_count').count().reset_index()

# Calculates various characteristics for the playgrounds data 
number=zip_code_distribution[zip_code_distribution['zip_code_count']>1]['id_clean'].sum()
number_three=zip_code_distribution[zip_code_distribution['zip_code_count']>2]['id_clean'].sum()
total_playgrounds=zip_code_distribution['id_clean'].sum()

percent="{0:.0%}".format(number/total_playgrounds)
percent_three="{0:.0%}".format(number_three/total_playgrounds)     ##-> Can be modularized
print (str(number)+" ("+percent+") playgrounds are associated with more then one zip code")
print (str(number_three)+" ("+percent_three+") playgrounds are associated with three or more zip codes")

number_of_schools=playgrounds[playgrounds['School_ID'].notnull()]['id_clean'].count()
percent_schools="{0:.0%}".format(number_of_schools/total_playgrounds)
print (str(number_of_schools)+" ("+percent_schools+") playgrounds are associated with a school and will use geocoding to determine zip code")

# Creates a distribution showing how many playgrounds have a specific number of zip code
dist=zip_code_distribution.plot.bar(x='zip_code_count',y='id_clean',title='Distribution of Associated Zip Code Count')
dist.set_xlabel('Number of Associated Zip Codes for a Playground')
dist.set_ylabel('Number of Playgrounds')


# This section of code takes the income data, cleans it and then calculates some metrics to bring it down to a zip code level. 
# 
# The data itself is income tax data for every zip code split between 6 Adjusted Gross Income (AGI) brackets. The end of this code outputs a file that has the zipcode to each one of that zipcode's metrics along with some descriptive statistics about the zipcode (such as st. dev which shows how economically diverse that zip code is)
# 
# Key: 
# 1. N1 = Total Number of Returns 
# 2. mars1 = Total Number of Single Returns 
# 3. MARS2 = Total Number of Joint Returns 
# 4. MARS 4 = Total Number of HoH Returns 
# 5. A00100 = Total Adjusted Gross Income for that Zip and Bracket

# In[256]:


### Income Tax Data Clean Up 

# Only getting NYC Zips from the main income file
ny_zips=ny_zips_raw[['County Name','ZIP Code']]
nyc_zips=ny_zips[ny_zips['County Name'].str.contains('|'.join(nyc_counties))] ## Filter for only NYC Counties

# Check to see which zips are missing from the income tax file
income_tax_check=income_tax_raw[['zipcode','agi_stub']].groupby('zipcode').sum().reset_index()
nyc_zip_check=nyc_zips.merge(income_tax_raw,left_on='ZIP Code',right_on='zipcode',how='left')
nyc_zip_missing=nyc_zip_check[nyc_zip_check['agi_stub'].isna()]['ZIP Code'].count()
nyc_zip_missing_percent=nyc_zip_missing/nyc_zip_check['ZIP Code'].count()
percent_format="{0:.0%}".format(nyc_zip_missing_percent)
print (str(nyc_zip_missing)+' ('+percent_format+') NYC Zip Codes are missing from the income tax file')

# Filter income data to only NYC data and join to get county
nyc_income=income_tax_raw[financials].merge(nyc_zips,left_on='zipcode',right_on='ZIP Code',how='inner')

# Calculate percent error between the total returns column and the sum of the different types of returns 
nyc_income['calc_total_returns']=nyc_income['mars1']+nyc_income['MARS2']+nyc_income['MARS4']
nyc_income['num_of_returns_error']=np.abs(nyc_income['N1']-nyc_income['calc_total_returns'])/nyc_income['N1']
percent_error="{0:.0%}".format((nyc_income['num_of_returns_error'].mean()))
print ("On average the sum of single,pair,HoH returns is off by "+percent_error )  ## Using number of returns total instead of summing all types of return because the difference is marginal

# Get some aggregates for each zip code
zip_income=nyc_income.groupby(['County Name','zipcode']).agg({'N1':['sum','max','mean','var','std'],
                                                             'MARS4':'sum'}).reset_index()

# Calculate a total agi for the HoH returns by getting the ratio of HoH returns to total
# MARS4 = Total Number of HoH returns 
# N1 = Total number of returns --> Using this instead of the sum of Mars1,Mars2,Mars4 introduces a slight bias 
# A00100 = Total AGI 
nyc_income['hoh_total_agi']=(nyc_income['MARS4']/nyc_income['N1'])*nyc_income['A00100']

# Calculates some preliminary stats at the bracket level. These stats are aggregated in different ways to 
# calculate a feature at the Zip Code Level 
nyc_income=nyc_income.merge(zip_income,on='zipcode',how='left')
nyc_income['weight']=nyc_income['N1']/nyc_income[('N1','sum')]
nyc_income['hoh_weight']=nyc_income['MARS4']/nyc_income[('MARS4','sum')]
nyc_income['median_return']=nyc_income[('N1','sum')]/2

## Calculating different ways to get a per capita wealth for a given zip code 
nyc_income['weighted_avg_agi']=nyc_income['A00100']*nyc_income['weight']    ## For weighted average of AGI by county
nyc_income['hoh_weighted_avg_agi']=nyc_income['hoh_total_agi']*nyc_income['hoh_weight']
nyc_income['individual_agi']=nyc_income['A00100']/nyc_income['N1']         ## Simple average of AGI by county (AGI/#of returns)  --> Needed for household income, compare to population in next version
nyc_income['mode_individual_agi']=nyc_income.apply(lambda row:find_mode_agi(row),axis=1 ) ## Gets the 'individual agi' for the bracket that has the most return count

# Calculating Median --> Need more efficient method for next version 
# The following snippet of code gets the median individual agi value based on the median number of returns 
returns_rolling_sum=nyc_income[['zipcode','agi_stub','N1']].groupby(by=['zipcode','agi_stub']).sum().groupby(level=[0]).cumsum().reset_index()
returns_rolling_sum=returns_rolling_sum.merge(nyc_income[['zipcode','median_return']],on='zipcode',how='left')
returns_rolling_sum['difference']=returns_rolling_sum.apply(lambda row: (row['N1']-row['median_return']) if (row['median_return']<row['N1'])else None,axis=1)
min_diff=returns_rolling_sum[['zipcode','difference']].groupby('zipcode').min()
returns_rolling_sum=returns_rolling_sum.merge(min_diff,on=['zipcode','difference'],how='inner').drop_duplicates().rename(columns={'agi_stub':'median_range'})  ## Really inefficient --> Need better design for production
nyc_income=nyc_income.merge(returns_rolling_sum[['zipcode','median_range']],on='zipcode',how='left')
nyc_income['median_individual_agi']=nyc_income.apply(lambda row: row['individual_agi'] if row['agi_stub']==row['median_range'] else None,axis=1)  ## Median Simple Average of a county (assumes there are only 5 differenct incomes in a county)


# Creation of data Set 1 ... Zip Code Financials by Averages 
# nyc_income.csv gives you statistics on a bracket level. It is used later in the script 
# whenever something needs to be calculated at a bracket level 

nyc_income.to_csv('nyc_bracket_level.csv')

# Aggregate individual values to get metric at Zip Level 
zip_income_averages=nyc_income.groupby(['County Name','zipcode']).agg({     #Need individual functions so that calculations can be modularized
    'weighted_avg_agi':'sum',
    'hoh_weighted_avg_agi':'sum', 
    'median_individual_agi':'max',
    'mode_individual_agi':'max',
    'A00100':['sum','mean'],
    'hoh_total_agi':'sum'
}).reset_index()

# Final Calculation of metrics
zip_income=zip_income.merge(zip_income_averages,on=['zipcode','County Name'],how='inner')
zip_income['weighted_avg_capita']=zip_income[('weighted_avg_agi','sum')]/zip_income[('N1','sum')]
zip_income['hoh_weighted_avg_capita']=zip_income[('hoh_weighted_avg_agi','sum')]/zip_income[('MARS4','sum')]
zip_income['simple_average']=zip_income[('A00100','sum')]/zip_income[('N1','sum')]
zip_income['hoh_simple_average']=zip_income[('MARS4','sum')]/zip_income[('hoh_total_agi','sum')]
zip_income['zipcode']=zip_income['zipcode'].astype('str')

# Creation of data set which provides all the income related stats and calculated metrics 
# for a given zip code
nyc_income.to_csv('nyc_income_zip_level.csv')


# This code visualizes the distribution of return types for the different counties 

# In[258]:


### Income Level Descriptions 

# print (list(nyc_income))

return_type=nyc_income[['County Name','zipcode','mars1','MARS2','MARS4']]
return_type_county=nyc_income[['County Name','mars1','MARS2','MARS4']].groupby('County Name').sum().rename(columns={'mars1':'Single Returns',
                                                                                                                   'MARS2':'Joint Returns',
                                                                                                                   'MARS4':'HoH Returns'}).reset_index()
# Calculates percentages for relational stacked bar                                                         
return_type_county['total_returns']=return_type_county['Single Returns']+return_type_county['Joint Returns']+return_type_county['HoH Returns']
return_type_county=return_type_county.sort_values('total_returns',ascending=False)
return_type_county['Single Return %']=return_type_county['Single Returns']/return_type_county['total_returns']*100
return_type_county['Joint Return %']=return_type_county['Joint Returns']/return_type_county['total_returns']*100
return_type_county['HoH Return %']=return_type_county['HoH Returns']/return_type_county['total_returns']*100

# Creates the visual
stacked_bar=return_type_county[['County Name','Single Returns','Joint Returns','HoH Returns']].plot(kind='bar',x='County Name',stacked=True)
stacked_bar.set_ylabel('Total Number of Returns')
percent_stacked_bar=return_type_county[['County Name','Single Return %','Joint Return %','HoH Return %']].plot(kind='bar',x='County Name',stacked=True)
percent_stacked_bar.set_ylabel('% of Total Returns')


# This section of code creates visualizations describing the calculated metrics

# In[259]:


### Population and Average Income Level Descriptions

population=zip_income[[('County Name',''),('zipcode',''),('N1','sum'),('N1','std')]]
population['normalized_scatter']=(population[('N1','std')]-population[('N1','std')].mean())/(population[('N1','std')].max()-population[('N1','std')].min())

pop_scatter_normal=population.plot.scatter(x=('N1','sum'),y='normalized_scatter')

# Creates multi colored scatter where each color is a different county
groups=population.groupby('County Name')
fig,ax=plt.subplots()
for name,group in groups:
    ax.plot(group[('N1','sum')],group[('normalized_scatter','')],marker='o',linestyle='',label=name)
ax.set_ylabel('Normalized Standard Dev. of Return Totals')
ax.set_xlabel('Total Number of Returns')
ax.set_title('The St. Dev in an Area\'s AGI Compared to Household Count')
ax.legend()
plt.show()

# Creates a line chart for each county showing the result of each metric 
# The x-axis is in order from lowest to highest standard deviation 
# So curve on the left represents the most homogenous zip codes going toward the least diverse 
zipcode_stats=zip_income[[('County Name',''),('zipcode',''),('N1','std'),('median_individual_agi','max'),('mode_individual_agi', 'max'),('weighted_avg_capita', ''),('simple_average', '')]]
zipcode_stats=zipcode_stats.sort_values(('N1','std'))

stats_bar=county_df[[('zipcode',''),('median_individual_agi','max'),('mode_individual_agi', 'max'),('weighted_avg_capita', ''),('simple_average', '')]]    .plot(kind='line',x=('zipcode',''),stacked=False,figsize=(10,5),title='Overall')
stats_bar.set_xlabel('Zipcode (in order of increasing standard deviation)')

for county in nyc_counties:
    county_df=zipcode_stats[zipcode_stats[('County Name','')]==county]
    county_df=county_df.sort_values(('N1','std'),ascending=True)
    plot_title='Household Income by zip code for '+county
    stats_bar=county_df[[('zipcode',''),('median_individual_agi','max'),('mode_individual_agi', 'max'),('weighted_avg_capita', ''),('simple_average', '')]]    .plot(kind='line',x=('zipcode',''),stacked=False,figsize=(10,5),title=plot_title)
    stats_bar.set_xlabel('Zipcode (in order of increasing standard deviation)')
    stats_bar.set_ylabel('Individual Household AGI')


# In[260]:


### Box and Whiskery Plots to show metrics 


box_and_whisker=zip_income[[('zipcode', ''),('median_individual_agi', 'max'),('mode_individual_agi', 'max'),('weighted_avg_capita', ''),('simple_average', '')]]
metrics=list(box_and_whisker)
metrics.remove(('zipcode', ''))
print (metrics)

for metric in metrics: 
    df=box_and_whisker[[('zipcode',''),metric]]
    baw_plot=df.boxplot(figsize=(10,10))
    plt.show()


# In this next section correlations between all the metrics and the number of playgrounds is calculated. No filters are placed in the data. The correlation can be outputted to a csv for further analysis 

# In[262]:


### Correlation of calculated metrics with playground count no filters

playground_count=playgrounds_by_zip[['Zip','id_clean']].groupby('Zip').count().reset_index().rename(columns={'id_clean':'playground_count'})              
playground_income_table=zip_income.merge(playground_count,left_on='zipcode',right_on='Zip',how='left')

playground_income_corr=correl_table(playground_income_table)

playground_income_corr.to_csv('income_playground_correl_no_filters.csv')

print (playground_income_corr)


# This section calculates a correlation between the number of returns for a given bracket and the parks. The purpose of this section is to identify any patterns with a particular income bracket. While no patterns across the city are discovered, each county has a special relationship with a given bracket. This is most likely a result of what kind of socio economic classes tend to live in these counties (This data result is not mentioned in the write up) 

# In[263]:


### Correlation of number of returns by tax bracket and playground count

agi_range=nyc_income[['County Name','zipcode','agi_stub','N1','MARS4']]
agi_range_pivot=agi_range.pivot_table(['N1','MARS4'],['County Name','zipcode'],'agi_stub').reset_index()

agi_range_pivot[('N1','sum')]=agi_range_pivot[[('N1', 1), ('N1', 2), ('N1', 3), ('N1', 4), ('N1', 5), ('N1', 6)]].sum(axis=1)
agi_range_pivot[('MARS4','sum')]=agi_range_pivot[[('MARS4', 1), ('MARS4', 2), ('MARS4', 3), ('MARS4', 4), ('MARS4', 5), ('MARS4', 6)]].sum(axis=1)

agi_range_pivot['zipcode']=agi_range_pivot['zipcode'].astype(str)
playground_agi_range=agi_range_pivot.merge(playground_count,left_on='zipcode',right_on='Zip',how='left')

agi_range_correl=correl_table(playground_agi_range)


agi_range.to_csv('correl_between_bracket_and_playgrounds.csv')

print (agi_range_correl)


# In this section, certain filters are placed on the input data to account for some outliers. The first is excluding large parks that are associated with multiple zip codes. The second is removing zip code outliers where the weighted average income is very large compared to the mean 

# In[266]:


### Repeating correlation calculation but excluding specific outliers


playgrounds_by_zip_clean=playgrounds_by_zip[playgrounds_by_zip['zip_code_count']<3]
playground_count_clean=playgrounds_by_zip_clean[['Zip','id_clean']].groupby('Zip').count().reset_index().rename(columns={'id_clean':'playground_count'})              

zip_income_clean=zip_income[zip_income[('weighted_avg_capita'),'']<=1000000] ## Filter on 14 to see effect but will affect visuals in next section 
playground_income_table_clean=zip_income_clean.merge(playground_count_clean,left_on='zipcode',right_on='Zip',how='left')
playground_income_corr=correl_table(playground_income_table_clean)

playground_agi_range_clean=agi_range_pivot.merge(playground_count_clean,left_on='zipcode',right_on='Zip',how='left')
playground_agi_range_corr=correl_table(playground_agi_range_clean)

playground_income_corr.to_csv('outliers_excluding_correlation.csv')

print (playground_income_corr)
print (playground_agi_range_corr)


# This section creates scatter visuals to show the relation of playground availability to Total HoH income and weighted average HoH Income

# In[270]:


### Income vs Playground Visuals --> on filtered data 

playground_income_visual=playground_income_table_clean[[('County Name',''),('zipcode',''),('hoh_total_agi','sum'),'playground_count']].rename(columns={('County Name',''):'County Name',('hoh_total_agi','sum'):'Total HoH Income' })

playground_income_visual_tmp=playground_income_visual[playground_income_visual['County Name']=='New York']

groups=playground_income_visual_tmp.groupby('County Name')
fig,ax=plt.subplots(figsize=(10,5))
for name,group in groups:
    ax.plot(group['playground_count'],group['Total HoH Income'],marker='o',linestyle='',label=name)
    ax.set_ylabel('Total HoH Income in Zip Code')
    ax.set_xlabel('Number of Available Playgrounds in Proximity')
    ax.set_title('Number of Playgrounds Available with Relation to Household Income (Manhattan Only)')
    ax.legend()
plt.show()

playground_income_visual_tmp=playground_income_visual[playground_income_visual['County Name']!='New York']

groups=playground_income_visual_tmp.groupby('County Name')
fig,ax=plt.subplots(figsize=(10,5))
for name,group in groups:
    ax.plot(group['playground_count'],group['Total HoH Income'],marker='o',linestyle='',label=name)
    ax.set_ylabel('Total HoH Income in Zip Code')
    ax.set_xlabel('Number of Available Playgrounds in Proximity')
    ax.set_title('Number of Playgrounds Available with Relation to Household Income (Excluding Manhattan)')
    ax.legend()
plt.show()

playground_income_visual=playground_income_table_clean[[('County Name',''),('zipcode',''),('hoh_weighted_avg_capita',''),'playground_count']].rename(columns={('County Name',''):'County Name',('hoh_weighted_avg_capita',''):'Weighted Average AGI' })

playground_income_visual_tmp=playground_income_visual[playground_income_visual['County Name']=='New York']

groups=playground_income_visual_tmp.groupby('County Name')
fig,ax=plt.subplots(figsize=(10,5))
for name,group in groups:
    ax.plot(group['playground_count'],group['Weighted Average AGI'],marker='o',linestyle='',label=name)
    ax.set_ylabel('Weighted Average AGI per Family')
    ax.set_xlabel('Number of Available Playgrounds in Proximity')
    ax.set_title('Number of Playgrounds Available with Relation to Weighted Average Household Income (Manhattan Only)')
    ax.legend()
plt.show()

playground_income_visual_tmp=playground_income_visual[playground_income_visual['County Name']!='New York']


groups=playground_income_visual_tmp.groupby('County Name')
fig,ax=plt.subplots(figsize=(10,5))
for name,group in groups:
    ax.plot(group['playground_count'],group['Weighted Average AGI'],marker='o',linestyle='',label=name)
    ax.set_ylabel('Weighted Average AGI per Family')
    ax.set_xlabel('Number of Available Playgrounds in Proximity')
    ax.set_title('Number of Playgrounds Available with Relation to Weighted Average Household Income (Excluding Manhattan)')
    ax.legend()
plt.show()


# The last three sections creates the map visual using three different metrics. Each map visual needs to be in its own section to run. 

# In[271]:


### Weighted Average AGI distribution 

w_avg_distribution=generate_map(zip_income,('zipcode',''),('hoh_weighted_avg_capita',''),'Weighted Average Income per Family')

w_avg_distribution


# In[272]:


### Total Family AGI Distribution in NYC

total_wealth_distribution=generate_map(zip_income,('zipcode',''),('hoh_total_agi','sum'),'Total Income from Families Distribution')

total_wealth_distribution


# In[273]:


### Playground Distribution in NYC 

playground_distribution=generate_map(playground_count_clean,'Zip','playground_count','Total Park Count In County')

playground_distribution


# Thank You for taking the time to read through this script!

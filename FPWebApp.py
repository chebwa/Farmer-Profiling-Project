import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import base64
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.write("""
# Farmer Profiling Web App

This app creates Tier profiles for **Small-scale Farmers** in a district through clustering them based on selected attributes- the starting point to understanding your farmers and making **informed** decisions or investments.

***
""")

colmn1= st.sidebar
colmn2, colmn3 = st.beta_columns((2,1))

colmn1.markdown("""
## File Upload
""")
colmn1.markdown("""
[Example CSV file template](https://raw.githubusercontent.com/chebwa/Sample-files/main/Example%20file.csv)
""")

uploaded_file=colmn1.file_uploader("Upload your input template file (.csv files only)", type= ["csv"])

colmn1.subheader("Farmer Profile Selection")
#read data and set the index
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    df= df.set_index(df.columns[0])
    

    #create new dataframe for the selected 3 columns(attributes) and rename columns to generic columns
    def Load_data():
    
        data=df[df.columns[0:3]]
        old_names=data.columns.values[0:3]
        new_names=['col1', 'col2', 'col3']
        data.rename(columns=dict(zip(old_names, new_names)), inplace=True)
        return data

    #Create statistics table to deal with outliers
    data=Load_data()
    data_stats= data.describe()
    data_stats=data_stats.T
    max_val= data_stats['max']
    qrt_75= data_stats['75%']
    data_stats['det']=qrt_75/max_val #Create determinant column to determine which outlier elimination method needs to be applied
    #Set the conditions and the elimination choices
    conditions= [(data_stats['det']<= 0.1),
        (data_stats['det']>0.1) & (data_stats['det']<= 0.75),
        (data_stats['det']>0.75)]
    choices=[(max_val-3*(data_stats['std'])),
        (max_val-data_stats['std']),
        (max_val)]
    #Apply the conditions and choices to new column called Cutoff
    data_stats['Cutoff']=np.select(conditions, choices)
    cutoff1= data_stats.loc['col1','Cutoff']
    cutoff2= data_stats.loc['col2','Cutoff']
    cutoff3= data_stats.loc['col3','Cutoff']
    #Create two seperate dataframes based on the outlier cutoffs
    outlier=data[(data['col1']>= cutoff1) | (data['col2']>= cutoff2) | (data['col3']>= cutoff3)]
    data_filtered=data[(data['col1']< cutoff1) & (data['col2']< cutoff2) & (data['col3']< cutoff3)]

    #Scale both datasets
    robust_scaler=RobustScaler()
    scaled_data= robust_scaler.fit_transform(data_filtered)
    scaled_outlier= robust_scaler.transform(outlier)

    #Use KMeans model to create clusters
    KMean_clust = KMeans(n_clusters= 3, init= 'k-means++', max_iter= 1000,  random_state=0)
    model= KMean_clust.fit(scaled_data)
    pickle.dump(model, open("kmeans_model.pkl", "wb"))

    loaded_model= pickle.load(open("kmeans_model.pkl", "rb"))
    loaded_model.fit(scaled_data)
    
    #Find the clusters for the observations given in the datasets and then combine the datasets
    filtered_clusters = loaded_model.fit_predict(scaled_data)
    outlier_clusters = loaded_model.predict(scaled_outlier)
    data_filtered['Cluster']=filtered_clusters
    outlier['Cluster']=outlier_clusters
    combined_data=data_filtered.append(outlier)
    df['Cluster']=combined_data['Cluster']

    #Create Scoring table to rank the clusters according to performance on each attribute
    Scoring_table=combined_data.groupby('Cluster')['col1', 'col2', 'col3'].mean()
    Scoring_table=Scoring_table.rank(axis=0, method='max')
    Scoring_table['Score']=Scoring_table['col1'] + Scoring_table['col2'] + Scoring_table['col3']

    #Assign Profile tiers to each Cluster
    def Profile(val):
        if val == Scoring_table['Score'].max():
        
            return 'Top Tier'
        elif val == Scoring_table['Score'].min():
            return 'Lower Tier'
        else:
            return 'Medium Tier'

    Scoring_table['Profile']= Scoring_table['Score'].apply(Profile)
    profile_dict= Scoring_table.to_dict()['Profile']

    def set_value(row_number, assigned_value):
        return assigned_value[row_number]

    #Create Profile column in df
    df['Profile']=df['Cluster'].apply(set_value, args=(profile_dict, ))

    
    df["Profile"].value_counts()
    
    unique_profiles= sorted(df.Profile.unique())
    selected_profile= colmn1.multiselect('Farmer Profile', unique_profiles, unique_profiles)
    df_selected= df[(df.Profile.isin(selected_profile))]

    colmn2.header('Display Farmer Data of Selected Profile(s)')
    colmn2.write('Filter the dataset by selecting the Farmer profile(s) you want displayed from the **Farmer Profile Selection** box.')
    colmn2.write('*Data Dimension: ' + str(df_selected.shape[0]) + ' rows and ' + str(df_selected.shape[1]) + ' columns.*')
    colmn2.dataframe(df_selected)

    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f' <a href="data:file/csv;base64,{b64}" download="selected_farmerprofile.csv">Download CSV File</a>'
        return href

    colmn2.markdown(filedownload(df_selected), unsafe_allow_html=True)

    colmn3.header('Farmer Profiles')
    colmn3.write("Explains the general characteristics of the Profiles created.")
    colmn3.write("""
    ##\n
    1. **Top Tier Farmer**- these are the role models in the district based on your chosen attributes, they have optimised practices and invested time in ensuring high performance.
    2. **Middle Tier Farmer**- these are the farmers with potential, they have made adequate strides in ensuring moderate to good performance compared to their peers but might need support to get them to the Top Tier.
    3. **Lower Tier Farmer**- these are farmers who may be struggling compared to their peers or in the case of Revenue-based attributes, they could be Subsistence farmers who farm for personal use.
    """)

    colmn2.header("**Farmer Profile Visualisation**")
    colmn2.write("""
    ##\n
    Select the attributes you want displayed for the *x* and *y* axes from the **Visualisation Attributes Selector** box.""")
    numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
    colmn1.subheader("Visualisation Attributes Selector")
    x_values= colmn1.selectbox('X axis', options =numeric_columns)
    y_values= colmn1.selectbox('Y axis', options =numeric_columns)
    plot = px.scatter(data_frame= df, x= x_values, y=y_values, color= 'Profile')
    colmn2.plotly_chart(plot)

    colmn3.write("""
    ##\n
    ##\n
    Select an attribute from the **Attributes Selector** box.""")
    cat_columns= list(df.select_dtypes(['O']).columns)
    colmn1.subheader("Attributes Selector")
    x1_values = colmn1.selectbox('Attribute', options = cat_columns)
    prof=df.groupby('Profile')[x1_values].value_counts().unstack()
    prof1= prof.T
    prof1.plot(kind="bar", stacked =True)
    plt.legend(loc= "lower left", bbox_to_anchor=(0.8,1.0))
    colmn3.pyplot(plt)
else:

    st.write("""
    **Instruction:** Please upload your filled in **input template (csv)**  in the sidebar on the left or download the **Example CSV file** and use it as an upload by right clicking the link and selecting ***Save link as*** then converting from a txt file to a csv file. 
    """)


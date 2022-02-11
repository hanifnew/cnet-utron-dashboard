from typing import Any, Dict, Tuple
from pathlib import Path
import toml
import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pickle
from datetime import datetime, timedelta
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import DistanceMetric
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn import metrics
import plotly.express as px
import ast
import plotly.graph_objects as go




# Set Page Config

st.set_page_config(page_title ="cnet-utron",
                    page_icon="🔮")#
# UNIX time 

def posix_time(dt):
                return (dt - datetime(1970, 1, 1)) / timedelta(seconds=1)

# Get Project Root

def get_project_root() -> str:
    
    return str(Path(__file__).parent)

# Load TOML config file

@st.cache(allow_output_mutation=True, ttl=300)
def load_config(
        config_readme_filename: str
) -> Dict[Any, Any]:

    config_readme = toml.load(Path(get_project_root()) / f"config/{config_readme_filename}")
    return dict(config_readme)

# Load config
readme = load_config("config_readme.toml")


####################################################################
######################## T I T L E #################################
####################################################################

st.title("Cellular Network User Throughput-Downlink Prediction Dashboard")

####################################################################
######################## A P P  I N F O ############################
####################################################################

with st.expander("🔎 What is this app?", expanded=False
):
    st.write("This app allows you to train, evaluate and optimize a Machine Learning model for **User Throughput Downlink Prediction** in just a few clicks. All you have to do is to upload a 4G LTE network  dataset from the **G-Net Track Application** or just choose the existing dataset in this app, and follow the guidelines in the sidebar to:")
    st.write("* 🧪  Prepare data : Filter, aggregate, preprocess, and/or clean your dataset step by step.")
    st.write("* 🛠️  Choose model parameters : Default parameters are available but you can tune them. Look at the tooltips to understand how each parameter is impacting forecasts.")
    st.write("* 📝  Select evaluation method : Gain the metrics evaluation score and Define the evaluation process to assess your model performance.")
    st.write("* 🔮  Generate forecast : Make a forecast on future throughput value with the model previously trained.")

df = pd.read_csv('dataset_existing/9am_2.csv',encoding='utf-7')
total_rows = df.shape[0]

########################################################################
######################## S I D E B A R #################################
########################################################################

image=Image.open('images/logo_1.png')
st.sidebar.image(image, use_column_width=True, width=100, caption='Cellular Network User Throughput-Downlink Prediction Dashboard')

#########################################################################
######################## D A T A  L O A D I N G #########################
#########################################################################

st.sidebar.subheader("Data Loading")
with st.sidebar.expander("Dataset"):

########################### U P L O A D  D A T A #########################

        data_upload = st.file_uploader("Upload a Clean Dataset", type=("txt"),
                        help=readme['tooltips']['data_upload'])

        if data_upload is not None:
                df = pd.read_csv(data_upload, delimiter = "\t",)
                df = df.rename(columns={
                        'Level' : 'RSRP',
                        'Qual' : 'RSRQ'
                })
                df = df[['Timestamp','Longitude','Latitude','Speed','Operator','CellID','LAC','LTERSSI','RSRP','RSRQ','SNR','DL_bitrate','UL_bitrate']]
                
                total_rows=df.shape[0]
                i=0
                while (i<(total_rows)):
                        df.Timestamp[i]=df.Timestamp[i][11:]
                        p = df.Timestamp[i]
                        a, b, c = p.partition('.')
                        df.Timestamp[i] = a + b.replace('.', '') + c.replace('.', '')
                        df.Timestamp[i]=df.Timestamp[i]
                        detik = df.Timestamp[i][4:]
                        menit = df.Timestamp[i][2:4]
                        jam = df.Timestamp[i][:2]
                        df.Timestamp[i]= int(jam)*3600+int(menit)*60+int(detik)
                        i+=1
                j=1
                while (j<total_rows):
                        df['Timestamp'][j] = df['Timestamp'][j] - df['Timestamp'][0]
                        j+=1
                df['Timestamp'][0] = 0
                df = df.fillna(0)
                df['SNR'] = pd.to_numeric(df['SNR'],errors='coerce')
                df['RSRP'] = pd.to_numeric(df['RSRP'],errors='coerce')
                df['LTERSSI'] = pd.to_numeric(df['LTERSSI'],errors='coerce')
                df['UL_bitrate'] = pd.to_numeric(df['UL_bitrate'],errors='coerce')
                df['RSRQ'] = pd.to_numeric(df['RSRQ'],errors='coerce')
                
                df = df.fillna(0)
                st.markdown('The **4G LTE Network** dataset is used as the example.')
                st.dataframe(df.head())
        
########################## E X I S T I N G  D A T A #########################

        if data_upload is None:
                existing_data = st.checkbox("Use Existing Dataset", value=False,
                                        help=readme['tooltips']['check_existing'])
                if existing_data :
                        data_builtin = st.selectbox("Select an Existing Dataset",
                                ['None','9am', '12pm','6pm'],
                                help=readme['tooltips']['existing_dataset'])
        
                        if data_builtin == '9am':
                                df = pd.read_csv('dataset_existing/9am_2.csv',encoding='utf-7')
                                df['Timestamp'] = pd.to_datetime(df['Timestamp']).apply(posix_time)
                                total_rows = df.shape[0]
                                i=1
                                while (i<total_rows):
                                        df['Timestamp'][i] = df['Timestamp'][i] - df['Timestamp'][0]
                                        i+=1
                                df['Timestamp'][0] = 0
                                df=df.fillna(0)
                                # st.markdown(df.SNR.dtypes)
                                st.markdown('The **Network Connectivity** dataset is used as the example.')
                                st.dataframe(df.head().astype(str))
                        elif data_builtin == '12pm':
                                df = pd.read_csv('dataset_existing/12pm_1.csv',encoding='utf-7')
                                df['Timestamp'] = pd.to_datetime(df['Timestamp']).apply(posix_time)
                                i=1
                                total_rows = df.shape[0]
                                while (i<total_rows):
                                        df['Timestamp'][i] = df['Timestamp'][i] - df['Timestamp'][0]
                                        i+=1
                                df['Timestamp'][0] = 0
                                df=df.fillna(0)
                                st.markdown('The **Network Connectivity** dataset is used as the example.')
                                st.dataframe(df.head().astype(str))
                        elif data_builtin == '6pm':
                                df = pd.read_csv('dataset_existing/6pm_1.csv',encoding='utf-7')
                                df['Timestamp'] = pd.to_datetime(df['Timestamp']).apply(posix_time)
                                total_rows = df.shape[0]
                                i=1
                                while (i<total_rows):
                                        df['Timestamp'][i] = df['Timestamp'][i] - df['Timestamp'][0]
                                        i+=1
                                df['Timestamp'][0] = 0
                                df=df.fillna(0)
                                st.markdown('The **Network Connectivity** dataset is used as the example.')
                                st.dataframe(df.head().astype(str))
                        else:
                                # df = None
                                st.warning('Please select a dataset!')
                        
                else :
                        st.warning('Please choose the dataset or upload in an Upload Form!')

                               

############ D A T E  A N D  T A R G E T   C O L U M N #################

with st.sidebar.expander("Column"):
        date_col = st.selectbox("Date column",
        ['Timestamp'], help=readme['tooltips']['date_column'])


        target_col = st.selectbox("Target column",
        ['DL_bitrate','Longitude','Latitude','Speed','Operator',
        'CellID','LAC','LTERSSI','RSRP','RSRQ','SNR',
        'UL_bitrate'], help=readme['tooltips']['target_column'])


#########################################################################
################## D A T A  P R E P A R A T I O N #######################
#########################################################################

st.sidebar.subheader("Data Preparation")
prep_method = st.sidebar.write("Select a preprocesssing method")


###################### F E A T U R E   S E L E C T I O N ##################

with st.sidebar.expander("Feature Selection"):
    show = st.checkbox("Features Correlation Heatmap", value=False, help=readme['tooltips']['heatmap'])
    if show:
        z = df[['Timestamp','Longitude','Latitude','Speed','Operator','CellID','LAC','LTERSSI','RSRP','RSRQ','SNR','DL_bitrate','UL_bitrate']].corr().round(2)
        fig = px.imshow(z)
        fig.update_layout(width=400, height=400)
        st.write(fig)
    dimensions_cols = st.multiselect(
            "Select dataset dimensions",
            ['Timestamp','Longitude','Latitude','Speed','Operator','CellID','LAC','LTERSSI','RSRP','RSRQ','SNR','DL_bitrate','UL_bitrate'],
            ['Timestamp', 'DL_bitrate','LTERSSI','RSRP','SNR'],
            help=readme['tooltips']['feature_selection']
        )
    temp = ""
    for col in dimensions_cols:
            temp = temp + "\'" + col + "\'"+ ","
    temp = temp[:-1]
    temp = "df = df[[" + temp + "]]"
    exec(temp)
    st.write(df.head())

    temp_5 = ""
    for col in dimensions_cols:
        if col != date_col and col != target_col:
            temp_5 = temp_5 + "\'" + col + "\'"+ ","
    temp_5 = temp_5[:-1]
    temp_5 = "numerical_features = [" + temp_5 + "]"
    exec(temp_5)

########### P R I N C I P A L  C O M P O N E N T  A N A L Y S I S ###########    

    perform_pca = st.checkbox("Perform PCA", value=False, help=readme['tooltips']['perform_pca'])
    if perform_pca:
            numerical_features = ['performance']
            num_pca = st.number_input(
                    'The minimum value is an integer of 3 or more.',
                        value=3, # default value
                        step=1,
                        min_value=2,
                        help=readme['tooltips']['error_message'])
            jumlah_kolom = len(dimensions_cols)
            if jumlah_kolom < 5 :
                    st.warning('We need at least 3 features without the date and the target column!')
            else :
                    pca = PCA(n_components=num_pca)
                    scaler = StandardScaler()

###### I N F O R M A T I O N  D I S T R I B U T I O N  (P C) ######

            temp_1 = ""
                
            for col in dimensions_cols:
                    if col != date_col and col != target_col:
                            temp_1 = temp_1 + "\'" + col + "\'"+ ","
            temp_1 = temp_1[:-1]
            temp_1 = "pca.fit(df[[" + temp_1 + "]])"
            exec(temp_1)
                
            hasil_pca = pca.explained_variance_ratio_
            st.caption('PCs Information Proportion of the PCA')
            st.write(hasil_pca)

########## D E T E R M I N E  P C A  D I M E N S I O N ############

            num_pca_2 = st.number_input(
                    'We need to set the number to 1 for reducing dimension to 1-dimension feature ',
                        value=1, # default value
                        step=1,
                        min_value=1,
                        help=readme['tooltips']['error_message'])
            pca_2 = PCA(n_components=num_pca_2)
            

            temp_2 = ""
            for col in dimensions_cols:
                    if col != date_col and col != target_col:
                            temp_2 = temp_2 + "\'" + col + "\'"+ ","
            temp_2 = temp_2[:-1]
            temp_2 = "pca_2.fit(df[[" + temp_2 + "]])"
            exec(temp_2)            

############# D I M E N S I O N  R E D U C T I O N  (P C A) ##########
                
            temp_3 = ""
            for col in dimensions_cols:
                    if col != date_col and col != target_col:
                            temp_3 = temp_3 + "\'" + col + "\'"+ ","
            temp_3 = temp_3[:-1]
            temp_3 = "df['performance'] = pca_2.transform(df.loc[:,(" + temp_3 + ")]).flatten()"
            exec(temp_3)
                
            temp_4 = ""
            for col in dimensions_cols:
                    if col != date_col and col != target_col:
                            temp_4 = temp_4 + "\'" + col + "\'"+ ","
            temp_4 = temp_4[:-1]
            temp_4 = "df.drop([" + temp_4 + "], axis=1, inplace=True)"
            exec(temp_4)
            st.write(df.head())

####################### T R A I N - T E S T  S P L I T ################

with st.sidebar.expander("Train-Test Split"):
        test_size = st.slider('% Size of test split:', min_value=1,max_value=99,value=30,
                        help=readme['tooltips']['train_test_split'])
        Table = df.set_index(date_col)
        training_data, testing_data = train_test_split(Table, test_size=0.01*test_size)
        x_train, y_train = training_data.drop(target_col, axis=1), training_data[target_col]
        x_test, y_test   = testing_data.drop(target_col, axis=1) , testing_data[target_col]

####################### S T A N D A R D I Z A T I O N ##################

scaler = StandardScaler()
scaler.fit(x_train[numerical_features])
x_train[numerical_features] = scaler.transform(x_train.loc[:, numerical_features])

#########################################################################
########################## M O D E L I N G ##############################
#########################################################################

st.sidebar.subheader("Modeling")
with st.sidebar.expander("Regression"):
        regressor = st.selectbox("Select a Regression Algorithm",   
                                 ['','K-Nearest Neighbors',
                                  'Random Forest', 'AdaBoost','Linear Regression',
                                  'Gradient Boosting','Support Vector Regression','Ridge Regression'],
                                  help=readme['tooltips']['regression_algorithm'])

if regressor :
        if regressor == 'K-Nearest Neighbors':
                st.sidebar.subheader('Hyperparameter Tuning')
                n_neighbors = st.sidebar.slider('k',value=5,min_value=1,max_value=20,help=readme['tooltips']['k_value'])
                metric = st.sidebar.selectbox("metric",
                                        ['minkowski','euclidean','manhattan',
                                        'chebyshev'],
                                        help=readme['tooltips']['metric_knn'])
                show_1 = st.checkbox("Generate Prediction", value=False)
                if show_1 :
                        #excecution time
                        ct0 = datetime.now(tz=None)
                        t0 = ct0.timestamp()
                        knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric)
                        knn.fit(x_train,y_train)
                        ct1 = datetime.now(tz=None)
                        t1=ct1.timestamp()
                        duration = t1-t0
                
                        x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])

                        # Evaluation Metrics
                        rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=knn.predict(x_test)))
                        r2_square = r2_score(y_true=y_test, y_pred=knn.predict(x_test))
                
                # Plot Prediction 
                
                        st.sidebar.subheader('Visualization')
                        viz = st.sidebar.radio(
                                "Choose Visualization Method",
                                ('st.line_chart', 'plotly'))
                        df_pred = pd.DataFrame(columns=['y_true','prediksi_KNN']) 
                        st.title('🕵️ Overview')
                        with st.expander("📉 More info on this plot", expanded=False):
                                st.write("This visualization displays several information:")
                                st.write("* There are two visualization method in this app, we use **st.line_chart** and **plotly line chart**")
                                st.write("* The blue line shows the **predictions** made by the model on both training and validation periods.")
                                st.write("* The orange line are the **actual values** of the target on training period.")
                                st.write("* In the st.line_chart viz, we can **scale up-down** and **shift left-right** the graph.")
                                st.write("* In the plotly viz, we can **see the future value by giving the difference between the true and predicted values, with a larger predicted value, using the slider**")
                                st.write(" You can also use the slider at the sidebar to range the period of data")

                        st.success(f"Training took {(duration*1000)} ms")

                        if viz == 'st.line_chart':
                        ###### Streamlit Line Chart ######
                                st.sidebar.subheader('Forecast Horizon')
                                horizon = st.sidebar.slider('Select range to predict',5,df.shape[0])
                                df_pred['y_true'] = y_test.iloc[:horizon]
                                df_pred['prediksi_KNN'] = knn.predict(x_test.iloc[:horizon].copy().round(1))
                                df_pred=df_pred.sort_values(by='Timestamp')
                                st.line_chart(df_pred)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                # with col3:
                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])
                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                    

                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")
                        else:
                        ##### Plotly ######
                                st.sidebar.subheader('Forecast Horizon')
                                time = st.sidebar.slider('true',1,int(df.shape[0]*(test_size*0.01)),value=80, help="Tune the true value")
                                horizon = st.sidebar.slider('prediction',1,int(df.shape[0]*(test_size*0.01)),value=100, help="Tune the predicted value")
                                fig, ax = plt.subplots()
                                ax.set_xlabel('Time')
                                ax.set_ylabel('Throughput')
                                ax.set_title('Throughput Prediction vs Actual')
                                ax.grid(True)
                                df_pred['y_true'] = y_test
                                df_pred['prediksi_KNN'] = knn.predict(x_test.copy().round(1))
                                df_pred=df_pred.sort_values(by='Timestamp')
                                # Plotting on the first y-axis
                                ax.plot(df_pred['prediksi_KNN'].iloc[:horizon], color='tab:cyan', label='Prediction')
                                ax.plot(df_pred['y_true'].iloc[:time], color='tab:orange', label='Actual')
                                ax.legend(loc='upper right')
                                st.plotly_chart(fig)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])

                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")


        elif regressor == 'Random Forest':
                st.sidebar.subheader('Hyperparameter Tuning')
                n_estimators = st.sidebar.slider('n_estimators', min_value=1, max_value=1000,value=100,help=readme['tooltips']['n_estimator'])
                show_2 = st.checkbox("Generate Prediction", value=False)
                if show_2 :
                        #excecution time
                        ct0 = datetime.now(tz=None)
                        t0 = ct0.timestamp()
                        RF = RandomForestRegressor(n_estimators=n_estimators, 
                                                random_state=55, n_jobs=-1)
                        RF.fit(x_train,y_train)
                        ct1 = datetime.now(tz=None)
                        t1=ct1.timestamp()
                        duration = t1-t0
                
                        x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])

                        # Evaluation Metrics
                        rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=RF.predict(x_test)))
                        r2_square= r2_score(y_true=y_test, y_pred=RF.predict(x_test))

                        st.sidebar.subheader('Visualization')
                        viz = st.sidebar.radio(
                                "Throughput Prediction Plot",
                                ('st.line_chart', 'plotly'))
                        df_pred = pd.DataFrame(columns=['y_true','prediksi_RF']) 

                        st.title('🕵️ Overview')
                        with st.expander("📉 More info on this plot", expanded=False):
                                st.write("This visualization displays several information:")
                                st.write("* There are two visualization method in this app, we use **st.line_chart** and **plotly line chart**")
                                st.write("* The blue line shows the **predictions** made by the model on both training and validation periods.")
                                st.write("* The orange line are the **actual values** of the target on training period.")
                                st.write("* In the st.line_chart viz, we can **scale up-down** and **shift left-right** the graph.")
                                st.write("* In the plotly viz, we can **see the future value by giving the difference between the true and predicted values, with a larger predicted value, using the slider**")
                                st.write(" You can also use the slider at the sidebar to range the period of data")
                        
                        st.success(f"Training took {(duration*1000)} ms")
                        
                        if viz == 'st.line_chart':
                        ###### Streamlit Line Chart ######
                                st.sidebar.subheader('Forecast Horizon')
                                horizon = st.sidebar.slider('Select range to predict',5,df.shape[0])
                                df_pred['y_true'] = y_test.iloc[:horizon]
                                df_pred['prediksi_RF'] = RF.predict(x_test.iloc[:horizon].copy().round(1))
                                df_pred=df_pred.sort_values(by='Timestamp')
                                st.line_chart(df_pred)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])
                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                
                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")
                        else:
                        ##### Plotly ######
                                st.sidebar.subheader('Forecast Horizon')
                                time = st.sidebar.slider('true',1,int(df.shape[0]*(test_size*0.01)),value=80, help="Tune the true value")
                                horizon = st.sidebar.slider('prediction',1,int(df.shape[0]*(test_size*0.01)),value=100, help="Tune the predicted value")
                                fig, ax = plt.subplots()
                                ax.set_xlabel('Time')
                                ax.set_ylabel('Throughput')
                                ax.set_title('Throughput Prediction vs Actual')
                                ax.grid(True)
                                df_pred['y_true'] = y_test
                                df_pred['prediksi_RF'] = RF.predict(x_test.copy().round(1))
                                df_pred=df_pred.sort_values(by='Timestamp')
                                # Plotting on the first y-axis
                                ax.plot(df_pred['y_true'].iloc[:time], color='tab:orange', label='Actual')
                                ax.plot(df_pred['prediksi_RF'].iloc[:horizon], color='tab:cyan', label='Prediction')
                                ax.legend(loc='upper right')
                                st.plotly_chart(fig)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")

        elif regressor == 'AdaBoost':
                st.sidebar.subheader('Hyperparameter Tuning')
                n_estimators = st.sidebar.slider('n_estimators', 1, 1000,value=50,help=readme['tooltips']['n_estimators_adaboost'])
                learning_rate = st.sidebar.slider('learning_rate', 0.01, 1.0,value=1.0,help=readme['tooltips']['learning_rate_adaboost'])
                show_3 = st.checkbox("Generate Prediction", value=False)
                if show_3 :
                        #excecution time
                        ct0 = datetime.now(tz=None)
                        t0 = ct0.timestamp()
                        boosting = AdaBoostRegressor(n_estimators=n_estimators, random_state=55, learning_rate=learning_rate)
                        boosting.fit(x_train,y_train)
                        ct1 = datetime.now(tz=None)
                        t1=ct1.timestamp()
                        duration = t1-t0

                        x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])

                        rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=boosting.predict(x_test)))
                        r2_square = r2_score(y_true=y_test, y_pred=boosting.predict(x_test))

                        st.sidebar.subheader('Visualization')
                        viz = st.sidebar.radio(
                                "Throughput Prediction Plot",
                                ('st.line_chart', 'plotly'))
                        df_pred = pd.DataFrame(columns=['y_true','prediksi_Boosting']) 
                        
                        st.title('🕵️ Overview')
                        with st.expander("📉 More info on this plot", expanded=False):
                                st.write("This visualization displays several information:")
                                st.write("* There are two visualization method in this app, we use **st.line_chart** and **plotly line chart**")
                                st.write("* The blue line shows the **predictions** made by the model on both training and validation periods.")
                                st.write("* The orange line are the **actual values** of the target on training period.")
                                st.write("* In the st.line_chart viz, we can **scale up-down** and **shift left-right** the graph.")
                                st.write("* In the plotly viz, we can **see the future value by giving the difference between the true and predicted values, with a larger predicted value, using the slider**")
                                st.write(" You can also use the slider at the sidebar to range the period of data")

                        st.success(f"Training took {(duration*1000)} ms")
                        
                        if viz == 'st.line_chart':
                        ###### Streamlit Line Chart ######
                                st.sidebar.subheader('Forecast Horizon')
                                horizon = st.sidebar.slider('Select range to predict',5,df.shape[0])
                                df_pred['y_true'] = y_test.iloc[:horizon]
                                df_pred['prediksi_boosting'] = boosting.predict(x_test.iloc[:horizon].copy().round(1))
                                df_pred=df_pred.sort_values(by='Timestamp')
                                st.line_chart(df_pred)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])
                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                    

                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")

                        else:
                        ##### Pyplot ######
                                st.sidebar.subheader('Forecast Horizon')
                                time = st.sidebar.slider('true',1,int(df.shape[0]*(test_size*0.01)),value=80, help="Tune the true value")
                                horizon = st.sidebar.slider('prediction',1,int(df.shape[0]*(test_size*0.01)),value=80, help="Tune the predicted value")
                                fig, ax = plt.subplots()
                                ax.set_xlabel('Time')
                                ax.set_ylabel('Throughput')
                                ax.set_title('Throughput Prediction vs Actual')
                                ax.grid(True)
                                df_pred['y_true'] = y_test
                                df_pred['prediksi_boosting'] = boosting.predict(x_test.copy().round(1))
                                df_pred=df_pred.sort_values(by='Timestamp')
                                # Plotting on the first y-axis
                                ax.plot(df_pred['y_true'].iloc[:time], color='tab:orange', label='Actual')
                                ax.plot(df_pred['prediksi_boosting'].iloc[:horizon], color='tab:cyan', label='Prediction')
                                ax.legend(loc='upper right')
                                st.plotly_chart(fig)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])

                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")

        
        elif regressor == 'Linear Regression':

                show_4 = st.checkbox("Generate Prediction", value=False)
                if show_4 :
                        #excecution time
                        ct0 = datetime.now(tz=None)
                        t0 = ct0.timestamp()
                        lin_reg = LinearRegression(normalize=True)
                        lin_reg.fit(x_train,y_train)
                        ct1 = datetime.now(tz=None)
                        t1=ct1.timestamp()
                        duration = t1-t0

                        x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])

                        rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=lin_reg.predict(x_test)))
                        r2_square= r2_score(y_true=y_test, y_pred=lin_reg.predict(x_test))

                        st.sidebar.subheader('Visualization')
                        viz = st.sidebar.radio(
                                "Throughput Prediction Plot",
                                ('st.line_chart', 'plotly'))
                        df_pred = pd.DataFrame(columns=['y_true','prediksi_linreg']) 

                        st.title('🕵️ Overview')
                        with st.expander("📉 More info on this plot", expanded=False):
                                st.write("This visualization displays several information:")
                                st.write("* There are two visualization method in this app, we use **st.line_chart** and **plotly line chart**")
                                st.write("* The blue line shows the **predictions** made by the model on both training and validation periods.")
                                st.write("* The orange line are the **actual values** of the target on training period.")
                                st.write("* In the st.line_chart viz, we can **scale up-down** and **shift left-right** the graph.")
                                st.write("* In the plotly viz, we can **see the future value by giving the difference between the true and predicted values, with a larger predicted value, using the slider**")
                                st.write(" You can also use the slider at the sidebar to range the period of data")


                        st.success(f"Training took {(duration*1000)} ms")

                        if viz == 'st.line_chart':
                        ###### Streamlit Line Chart ######
                                st.sidebar.subheader('Forecast Horizon')
                                horizon = st.sidebar.slider('Select range to predict',5,50)
                                df_pred['y_true'] = y_test.iloc[:horizon]
                                df_pred['prediksi_linreg'] = lin_reg.predict(x_test.iloc[:horizon].copy().round(1))
                                
                                df_pred=df_pred.sort_values(by='Timestamp')
                                st.line_chart(df_pred)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])
                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                    

                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")
                        else:
                        ##### Pyplot ######
                                st.sidebar.subheader('Forecast Horizon')
                                time = st.sidebar.slider('true',1,int(df.shape[0]*(test_size*0.01)),value=80, help="Tune the true value")
                                horizon = st.sidebar.slider('prediction',1,int(df.shape[0]*(test_size*0.01)),value=100, help="Tune the predicted value")
                                fig, ax = plt.subplots()
                                ax.set_xlabel('Time')
                                ax.set_ylabel('Throughput')
                                ax.set_title('Throughput Prediction vs Actual')
                                ax.grid(True)
                                df_pred['y_true'] = y_test
                                df_pred['prediksi_linreg'] = lin_reg.predict(x_test.copy().round(1))
                                df_pred=df_pred.sort_values(by='Timestamp')
                                # Plotting on the first y-axis
                                ax.plot(df_pred['y_true'].iloc[:time], color='tab:orange', label='Actual')
                                ax.plot(df_pred['prediksi_linreg'].iloc[:horizon], color='tab:cyan', label='Prediction')
                                ax.legend(loc='upper right')
                                st.plotly_chart(fig)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])

                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")

        elif regressor == 'Gradient Boosting':
                st.sidebar.subheader('Hyperparameter Tuning')
                n_estimators = st.sidebar.slider('n_estimators', 1, 1000, value=100,help=readme['tooltips']['n_estimators_gboost'])
                learning_rate = st.sidebar.slider('learning_rate', 0.01, 0.50, value=0.1, help=readme['tooltips']['learning_rate_gboost'])
                max_depth = st.sidebar.number_input("max_depth",
                        value=3, # default value
                        step=1,
                        min_value=1,
                        help=readme['tooltips']['max_depth_gboost'])
                loss = st.sidebar.selectbox('loss',
                                        ['squared_error', 'absolute_error', 'huber', 'quantile'],
                                        help=readme['tooltips']['loss_gboost'])
                show_5 = st.checkbox("Generate Prediction", value=False)
                if show_5 :
                        #excecution time
                        ct0 = datetime.now(tz=None)
                        t0 = ct0.timestamp()
                        gboosting = GradientBoostingRegressor(n_estimators=n_estimators, random_state=55)
                        gboosting.fit(x_train,y_train)  
                        ct1 = datetime.now(tz=None)
                        t1=ct1.timestamp()
                        duration = t1-t0

                        x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])

                        rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=gboosting.predict(x_test)))
                        r2_square = r2_score(y_true=y_test, y_pred=gboosting.predict(x_test))

                        st.sidebar.subheader('Visualization')
                        viz = st.sidebar.radio(
                                "Throughput Prediction Plot",
                                ('st.line_chart', 'plotly'))
                        df_pred = pd.DataFrame(columns=['y_true','prediksi_gBoosting'])

                        st.title('🕵️ Overview')
                        with st.expander("📉 More info on this plot", expanded=False):
                                st.write("This visualization displays several information:")
                                st.write("* There are two visualization method in this app, we use **st.line_chart** and **plotly line chart**")
                                st.write("* The blue line shows the **predictions** made by the model on both training and validation periods.")
                                st.write("* The orange line are the **actual values** of the target on training period.")
                                st.write("* In the st.line_chart viz, we can **scale up-down** and **shift left-right** the graph.")
                                st.write("* In the plotly viz, we can **see the future value by giving the difference between the true and predicted values, with a larger predicted value, using the slider**")
                                st.write(" You can also use the slider at the sidebar to range the period of data")

                        st.success(f"Training took {(duration*1000)} ms") 

                        if viz == 'st.line_chart':
                        ###### Streamlit Line Chart ######
                                st.sidebar.subheader('Forecast Horizon')
                                horizon = st.sidebar.slider('Select range to predict',5,50)
                                df_pred['y_true'] = y_test.iloc[:horizon]
                                df_pred['prediksi_gBoosting'] = gboosting.predict(x_test.iloc[:horizon].copy().round(1))
                                df_pred=df_pred.sort_values(by='Timestamp')
                                st.line_chart(df_pred)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])
                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                    

                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")
                        else:
                        ##### Pyplot ######
                                st.sidebar.subheader('Forecast Horizon')
                                time = st.sidebar.slider('true',1,int(df.shape[0]*(test_size*0.01)),value=80, help="Tune the true value")
                                horizon = st.sidebar.slider('prediction',1,int(df.shape[0]*(test_size*0.01)),value=100, help="Tune the predicted value")
                                fig, ax = plt.subplots()
                                ax.set_xlabel('Time')
                                ax.set_ylabel('Throughput')
                                ax.set_title('Throughput Prediction vs Actual')
                                ax.grid(True)
                                df_pred['y_true'] = y_test
                                df_pred['prediksi_gBoosting'] = gboosting.predict(x_test.copy().round(1))
                                df_pred=df_pred.sort_values(by='Timestamp')
                                # Plotting on the first y-axis
                                ax.plot(df_pred['y_true'].iloc[:time], color='tab:orange', label='Actual')
                                ax.plot(df_pred['prediksi_gBoosting'].iloc[:horizon], color='tab:cyan', label='Prediction')
                                ax.legend(loc='upper right')
                                st.plotly_chart(fig)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])

                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")
        
        elif regressor == 'Support Vector Regression':
                st.sidebar.subheader('Hyperparameters Tuning')
                epsilon = st.sidebar.number_input(
                    'epsilon',
                        value=0.10, # default value
                        step=0.01,
                        help=readme['tooltips']['epsilon'])
                C = st.sidebar.slider('C', 1, 10000, help=readme['tooltips']['C'])
                show_6 = st.checkbox("Generate Prediction", value=False)
                if show_6 :
                        #excecution time
                        ct0 = datetime.now(tz=None)
                        t0 = ct0.timestamp()
                        svm = LinearSVR(epsilon=epsilon, C=C)
                        svm.fit(x_train,y_train) 
                        ct1 = datetime.now(tz=None)
                        t1=ct1.timestamp()
                        duration = t1-t0 

                        x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])

                        rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=svm.predict(x_test)))
                        r2_square = r2_score(y_true=y_test, y_pred=svm.predict(x_test))

                        st.sidebar.subheader('Visualization')
                        viz = st.sidebar.radio(
                                "Throughput Prediction Plot",
                                ('st.line_chart', 'plotly'))
                        df_pred = pd.DataFrame(columns=['y_true','prediksi_svm']) 

                        st.title('🕵️ Overview')
                        with st.expander("📉 More info on this plot", expanded=False):
                                st.write("This visualization displays several information:")
                                st.write("* There are two visualization method in this app, we use **st.line_chart** and **plotly line chart**")
                                st.write("* The blue line shows the **predictions** made by the model on both training and validation periods.")
                                st.write("* The orange line are the **actual values** of the target on training period.")
                                st.write("* In the st.line_chart viz, we can **scale up-down** and **shift left-right** the graph.")
                                st.write("* In the plotly viz, we can **see the future value by giving the difference between the true and predicted values, with a larger predicted value, using the slider**")
                                st.write(" You can also use the slider at the sidebar to range the period of data")

                        st.success(f"Training took {(duration*1000)} ms")

                        if viz == 'st.line_chart':
                        ###### Streamlit Line Chart ######
                                st.sidebar.subheader('Forecast Horizon')
                                horizon = st.sidebar.slider('Select range to predict',5,50)
                                df_pred['y_true'] = y_test.iloc[:horizon]
                                df_pred['prediksi_svm'] = svm.predict(x_test.iloc[:horizon].copy().round(1))
                                df_pred=df_pred.sort_values(by='Timestamp')
                                st.line_chart(df_pred)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])
                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                    

                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")
                        else:
                        ##### Pyplot ######
                                st.sidebar.subheader('Forecast Horizon')
                                time = st.sidebar.slider('true',1,int(df.shape[0]*(test_size*0.01)),value=80, help="Tune the true value")
                                horizon = st.sidebar.slider('prediction',1,int(df.shape[0]*(test_size*0.01)),value=100, help="Tune the predicted value")
                                fig, ax = plt.subplots()
                                ax.set_xlabel('Time')
                                ax.set_ylabel('Throughput')
                                ax.set_title('Throughput Prediction vs Actual')
                                ax.grid(True)
                                df_pred['y_true'] = y_test
                                df_pred['prediksi_svm'] = svm.predict(x_test.copy().round(1))
                                df_pred=df_pred.sort_values(by='Timestamp')
                                # Plotting on the first y-axis
                                ax.plot(df_pred['y_true'].iloc[:time], color='tab:orange', label='Actual')
                                ax.plot(df_pred['prediksi_svm'].iloc[:horizon], color='tab:cyan', label='Prediction')
                                ax.legend(loc='upper right')
                                st.plotly_chart(fig)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])

                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")

        elif regressor == 'Ridge Regression':
                alpha = st.sidebar.number_input(
                    'alpha',
                        value=1.00, # default value
                        step=0.01,
                        help=readme['tooltips']['epsilon'])
                show_8 = st.checkbox("Generate Prediction", value=False)
                if show_8 :
                        #excecution time
                        ct0 = datetime.now(tz=None)
                        t0 = ct0.timestamp()
                        rr = Ridge(alpha=alpha)
                        rr.fit(x_train,y_train)
                        ct1 = datetime.now(tz=None)
                        t1=ct1.timestamp()
                        duration = t1-t0

                        x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])

                        rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=rr.predict(x_test)))
                        r2_square= r2_score(y_true=y_test, y_pred=rr.predict(x_test))

                        st.sidebar.subheader('Visualization')
                        viz = st.sidebar.radio(
                                "Throughput Prediction Plot",
                                ('st.line_chart', 'plotly'))
                        df_pred = pd.DataFrame(columns=['y_true','prediksi_rr']) 

                        st.title('🕵️ Overview')
                        with st.expander("📉 More info on this plot", expanded=False):
                                st.write("This visualization displays several information:")
                                st.write("* There are two visualization method in this app, we use **st.line_chart** and **plotly line chart**")
                                st.write("* The blue line shows the **predictions** made by the model on both training and validation periods.")
                                st.write("* The orange line are the **actual values** of the target on training period.")
                                st.write("* In the st.line_chart viz, we can **scale up-down** and **shift left-right** the graph.")
                                st.write("* In the plotly viz, we can **see the future value by giving the difference between the true and predicted values, with a larger predicted value, using the slider**")
                                st.write(" You can also use the slider at the sidebar to range the period of data")

                        st.success(f"Training took {(duration*1000)} ms")

                        if viz == 'st.line_chart':
                        ###### Streamlit Line Chart ######
                                st.sidebar.subheader('Forecast Horizon')
                                horizon = st.sidebar.slider('Select range to predict',5,df.shape[0])
                                df_pred['y_true'] = y_test.iloc[:horizon]
                                df_pred['prediksi_rr'] = rr.predict(x_test.iloc[:horizon].copy().round(1))
                                
                                df_pred=df_pred.sort_values(by='Timestamp')
                                st.line_chart(df_pred)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))

                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])
                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                    

                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")
                        else:
                        ##### Pyplot ######
                                st.sidebar.subheader('Forecast Horizon')
                                time = st.sidebar.slider('true',1,int(df.shape[0]*(test_size*0.01)),value=80, help="Tune the true value")
                                horizon = st.sidebar.slider('prediction',1,int(df.shape[0]*(test_size*0.01)),value=100, help="Tune the predicted value")
                                fig, ax = plt.subplots()
                                ax.set_xlabel('Time')
                                ax.set_ylabel('Throughput')
                                ax.set_title('Throughput Prediction vs Actual')
                                ax.grid(True)
                                df_pred['y_true'] = y_test
                                df_pred['prediksi_rr'] = rr.predict(x_test.copy().round(1))
                                df_pred=df_pred.sort_values(by='Timestamp')
                                # Plotting on the first y-axis
                                ax.plot(df_pred['y_true'].iloc[:time], color='tab:orange', label='Actual')
                                ax.plot(df_pred['prediksi_rr'].iloc[:horizon], color='tab:cyan', label='Prediction')
                                ax.legend(loc='upper right')
                                st.plotly_chart(fig)

                                st.header('🧮Performance Metrics')
                                col1, col2 = st.columns(2)
                                with col1 :
                                        st.metric("R2 Score","{:.2f}".format(r2_square))
                                with col2 :
                                        st.metric("RMSE","{:.2f}".format(rmse))
                                st.write("Tips on the Evaluation Metrics 💡")
                                st.info(readme['tooltips']['eval_tips'])

                                with st.expander("⚡ More info on Evaluation Metrics", expanded=False):
                                        st.write("The following metrics can be computed to evaluate model performance:")
                                        st.write("* __Root Mean Squared Error (RMSE)__: Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
                                        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                                        st.write("* __R2(squared) Score__: measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. It also called the coefficient of determination.")
                                        st.latex(r"R^2 Score = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Truth_i - \overline{Truth_i})^2}")
                                        
                                with st.expander("✔️  How to evaluate my model?", expanded=False):
                                        st.write("The following metrics and plots allow you to evaluate model performance. Go to the Sidebar if you wish to customize evaluation settings by:")
                                        st.write("* **Adding or Removing features in dataframe**")
                                        st.write("* **Comparing Result using PCA method with not using it**")
                                        st.write("* **Increasing or Reducing the number of PC in PCA section**")
                                        st.write("* **Tuning the Train-Test Data Proportion and Model Hyperparameters**")
                                        st.write("* **Changing the forecast period on Forecast Horizon**")
                                        st.write("You can also compare each model performance manually to find **the best machine learning model configuration** ")





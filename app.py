from json import encoder
# from altair.vegalite.v4.api import value
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import os
import pandas as pd
import sklearn
import time
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from scapy.all import sniff
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import plotly.express as px
import paramiko
import s3fs





st.title('Network Anomaly Detection Workbench with Machine Learning and Real-Time Packet Analysis')

st.write("""
#
""")



hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)




#Defining Python Global Class Design Pattern here
class Dataframe:
    """
    Global class to hold shared data structures for the application, particularly dataframes.
    """
    ratio=0.33

class MetricsReport:
    """
    Global class to hold and manage metrics reports generated during the application's runtime.
    """
    report = []

#Defining Python Strategy Design Pattern here
class StrategyClass:
    """
    Class representing a strategy pattern allowing for different execution strategies to be injected dynamically.
    """
    def __init__(self, func=None):
        """
        Initializes the strategy class. Optionally accepts a function to use as the execution strategy.
        
        Args:
            func: A function to be used as the strategy for execution. If None, a default message is printed.
        """
        
        if func:
             self.execute = func

    def execute(self):
        """
        Default execution method, prints a message indicating no specific strategy was provided.
        """
        
        print("No Execution Passed to Strategy Class")

class DBConnection:
    """
    Class providing static methods for database connections and data retrieval.
    """
    @staticmethod
    @st.cache(ttl=600, suppress_st_warning=True)
    def get_file_from_bucket(file_path):
        """
        Retrieves a file from an AWS S3 bucket using the provided file path.

        Args:
            file_path (str): The path to the file in the S3 bucket.

        Returns:
            pd.DataFrame: The data from the S3 file loaded into a pandas dataframe.
        """
        
        dfs3 = pd.read_csv(file_path, storage_options={"anon": False})
        return dfs3

    @staticmethod
    def fetchDB():
        """
        Interactive method using Streamlit to select and fetch data from an S3 bucket.

        Returns:
            uploaded_file: The file fetched from the database, or None if not fetched.
        """
        
        uploaded_file = None
        fs = s3fs.S3FileSystem(anon=False)
        buckets = fs.ls('/')
        selected_bucket = st.selectbox('Select AWS S3 Bucket: ', buckets)

        bucket_path = 's3://'+str(selected_bucket)+'/'
        dataset_names = fs.ls(bucket_path)
        dataset = st.selectbox('Select Dataset: ', dataset_names)

        connect_btn = st.checkbox("Use selected Dataset file from Database")
        if connect_btn:
            file_path = 's3://'+str(dataset)
            uploaded_file = DBConnection.get_file_from_bucket(file_path)
               
        return uploaded_file


class PacketCapture:
    """
    Class providing methods to perform local packet captures and process the captured data.
    """
    
    @staticmethod
    def local_packet_capture(capture_duration=15):
        """
        Captures packets on the local machine using Scapy for a specified duration.

        Args:
            capture_duration (int): Duration in seconds for which to capture packets. Defaults to 15.
        """
        
        packets = sniff(timeout=capture_duration)
        
        # Process packets and save to a .txt file 
        with open('packet_capture.txt', 'w') as packet_capture_file:
            for packet in packets:
                packet_capture_file.write(packet.summary() + '\n')

        # Convert captured packets to CSV format 
        PacketCapture.convert_packets_to_csv('packet_capture.txt', 'packet_capture.csv')
        PacketCapture.add_live_capture_labels()
        
    @staticmethod
    def convert_packets_to_csv(input_txt, output_csv):
        """
        Converts the packet data from a .txt file to a .csv file.

        Args:
            input_txt (str): The input .txt file containing packet data.
            output_csv (str): The output .csv file for converted data.
        """
        # Read from .txt file
        read_file = pd.read_csv(input_txt, delimiter="\t")
        
        # Convert to .csv file
        read_file.to_csv(output_csv, index=None)

    @staticmethod
    def add_live_capture_labels():
        """
        Adds descriptive labels to the columns of the live packet capture CSV file.
        """
        
        live_capture_labels = {0: 'duration', 1: 'protocol_type', 2: 'service', 3: 'flag', 4: 'src_bytes', 5: 'dst_bytes', 6: 'land', 7: 'wrong_fragment', 8: 'urgent', 9: 'count', 10: 'srv_count', 11: 'serror_rate', 12: 'srv_serror_rate', 13: 'rerror_rate', 14: 'srv_rerror_rate', 15: 'same_srv_rate', 16: 'diff_srv_rate', 17: 'srv_diff_host_rate', 18: 'dst_host_count', 19: 'dst_host_srv_count', 20: 'dst_host_same_srv_rate', 21: 'dst_host_diff_srv_rate', 22: 'dst_host_same_src_port_rate', 23: 'dst_host_srv_diff_host_rate', 24: 'dst_host_serror_rate', 25: 'dst_host_srv_serror_rate', 26: 'dst_host_rerror_rate', 27: 'dst_host_srv_rerror_rate'}
        Dataframe.X_test_live_packets = pd.read_csv('packet_capture.csv', header=None)
        Dataframe.X_test_live_packets.rename(columns= live_capture_labels, inplace=True)
        Dataframe.X_test_live_packets.to_csv('packet_capture.csv', index=False)
        Dataframe.X = Dataframe.X_test_live_packets.iloc[:, :-1]  # Assuming last column is the target
        Dataframe.y = Dataframe.X_test_live_packets.iloc[:, -1]

        st.subheader("Head of captured packets dataframe:")
        st.write(Dataframe.X_test_live_packets.head(10))

    
    @staticmethod
    def populate_live_capture():
        """
        Streamlit form for capturing packet duration and initiating the packet capture.
        """
        capture_duration = st.number_input("Duration of live packet capture in seconds", min_value=1, max_value=9999, value=15)
        start_capture_btn = st.button("Start Packet Capture")
        if start_capture_btn:
            PacketCapture.local_packet_capture(capture_duration=capture_duration)



class DataExploration:
    """
    Class for performing data exploration tasks such as plotting distributions of columns,
    providing dataset statistics, and generating scatter plots.
    """
    
    def plot_column( col ):
        """
        Plots the distribution of a specified column. If the column is categorical, it is encoded
        with LabelEncoder before plotting.

        Args:
            col (str): The column in the dataframe to plot.
        """
        if Dataframe.df[col].dtype == 'object':
            encoder=LabelEncoder()
            Dataframe.df[col] = encoder.fit_transform(Dataframe.df[col])
        fig, ax = plt.subplots()
        Dataframe.df.hist(
            bins=8,
            column= col ,
            grid=False,
            figsize=(8, 8),
            color="#86bf91",
            zorder=2,
            rwidth=0.9,
            ax=ax,
        )
        st.write(fig)

    #@st.cache(suppress_st_warning=True)
    def write_statistics(statistics, visualizaitons):
        """
        Displays selected statistics and visualizations based on user input.

        Args:
            statistics (list): A list of statistics the user wants to view.
            visualizations (list): A list of visualizations the user wants to view.
        """
        
        if 'Dataset Shape' in statistics:
            st.write('Shape of Dataset:', Dataframe.df.shape)
        if 'Number of Classes' in statistics:
            st.write('Number of Classes:', len(np.unique(Dataframe.y)))
        if 'Dataset Head' in statistics:
            st.write('Dataset Head:', Dataframe.df.head(5))
        if 'Describe Features' in statistics:
            st.write('Feature Description:', Dataframe.df.describe())
        if 'View Packet Types' in statistics:
            st.write('Packet Types:', np.unique(Dataframe.y))
        if 'Scatter Plots' in statistics:
            st.subheader("Scatter Plot:")
            plot_dim = st.selectbox("Select plot dimensionality", ('2D Plot', '3D Plot'))
            with st.form('Scatter Plot Form'):
                max_rows = min(200, len(Dataframe.df.index))
                num_samples = st.slider(label="Select number of random samples", min_value=1, max_value=max_rows)
                sampling_technique = st.radio('Sampling Technique:', ['Random Sampling', 'Equal Distribution Sampling'])
                if sampling_technique == 'Random Sampling':
                    sample_df = Dataframe.df.sample(num_samples)
                else:
                    sample_df = Dataframe.df.groupby(Dataframe.df.columns[len(Dataframe.df.columns)-1]).apply(lambda x: x.sample(num_samples, replace=True))

                if plot_dim == '2D Plot':
                    feature_x = st.selectbox('Select X-Axis Feature', (Dataframe.df.columns))
                    feature_y = st.selectbox('Select Y-Axis Feature', (Dataframe.df.columns))
                    if(feature_x and feature_y):
                        fig = px.scatter(sample_df, x= feature_x, y = feature_y, color=sample_df.columns[len(sample_df.columns)-1])
                        st.plotly_chart(fig)
                if plot_dim == '3D Plot':
                    feature_x = st.selectbox('Select X-Axis Feature', (Dataframe.df.columns))
                    feature_y = st.selectbox('Select Y-Axis Feature', (Dataframe.df.columns))
                    feature_z = st.selectbox('Select Z-Axis Feature', (Dataframe.df.columns))
                    if(feature_x and feature_y):
                        fig = px.scatter_3d(sample_df, feature_x, feature_y, feature_z, color = sample_df.columns[len(sample_df.columns)-1])
                        st.plotly_chart(fig)

                scatter_submit = st.form_submit_button('Apply Selected Options')

            

        if visualizaitons:
            DataExploration.write_visualizations(visualizaitons)

    #@st.cache(suppress_st_warning=True)
    def write_visualizations(visualizaitons):
        """
        Generates visualizations for each column selected by the user.

        Args:
            visualizations (list): A list of columns the user wants to visualize.
        """
        for column in visualizaitons:
            DataExploration.plot_column(col=column)

    
    def populate_statistics():
        """
        Renders the sidebar for the data exploration section in the UI, allowing users to select
        various statistics and visualizations they want to generate.
        """
        
        st.sidebar.header('Data Exploration')
        #Print statistics and visualizations sidebar items
        statistics=False
        visualizaitons=False
        with st.sidebar.form('Statistics Form'):
            statistics = st.multiselect(
                'Select Desired Statistics',
                ('Dataset Head', 'Dataset Shape', 'Number of Classes', 'Describe Features', 'View Packet Types', 'Scatter Plots', 'Plot Feature Visualizations')
            )
            statistics_submit = st.form_submit_button('Show Selected Options')

        if 'Plot Feature Visualizations' in statistics:
            with st.sidebar.form('Visualizations Form'):
                visualizaitons = st.multiselect(
                    'Select Desired Visualizations',
                    (Dataframe.df.columns)
                )
                visualizations_submit = st.form_submit_button('Show Selected Options')

        if statistics:
            DataExploration.write_statistics(statistics, visualizaitons)


class DataInput:
    """
    Class for handling data input, including reading data from various sources, preprocessing,
    and preparing it for further analysis.
    """
    def read_data():
        """
        Reads data from the specified source, preprocesses it by filling NaN values,
        replacing infinite values, splitting into features and labels, and encoding categorical variables.
        """
        #Reading uploading dataset csv file here
        if prediction_type == 'Connect to AWS S3 Database':
            Dataframe.df = uploaded_file
        else:
            Dataframe.df = pd.read_csv(uploaded_file)


        #Replace NaN/Infinite values with 0
        Dataframe.df = Dataframe.df.fillna(0)
        Dataframe.df = Dataframe.df.replace([np.inf, -np.inf], 0)

        #Splitting x & y dataframes here
        Dataframe.y = Dataframe.df.iloc[:,[-1]]
        Dataframe.X = Dataframe.df.iloc[: , :-1]

        #Label encoding categorical features here
        encoder = LabelEncoder()
        num_cols = Dataframe.X._get_numeric_data().columns
        cate_cols = list(set(Dataframe.X.columns)-set(num_cols))
        for item in cate_cols:
            Dataframe.X[item] = encoder.fit_transform(Dataframe.X[item])


        #Displaying dataset statistics and visualizaitons here
        strategy = StrategyClass(DataExploration.populate_statistics)
        strategy.execute()
        strategy = StrategyClass(Preprocessor.populate_preprocessors)
        strategy.execute()


class Preprocessor:
    """
    Class for handling data preprocessing tasks such as null value handling, scaling, and
    setting train-test split ratios.
    """
    def populate_preprocessors():
        """
        Renders the preprocessing options in the sidebar of the UI, allowing users to
        select and apply various preprocessing techniques to the data.
        """
        st.sidebar.header('Preprocessing')

        #Drop null values here:
        drop_nulls_btn = st.sidebar.checkbox('Drop Rows with Null Values')
        if drop_nulls_btn:
            Dataframe.df = Dataframe.df.dropna(axis=0)
        
        #Print preprocessing sidebar items
        scaling_btn = st.sidebar.checkbox('Apply Logarithmic Scaling')
        if scaling_btn:
            #Applying logarithmic scaling here
            sc = MinMaxScaler()
            Dataframe.X[Dataframe.X.columns] = sc.fit_transform(Dataframe.X)

        ratio_btn = st.sidebar.selectbox('Select Custom/Default Test-Train Ratio',('Default', 'Custom'))
        if ratio_btn == 'Default':
            Dataframe.ratio = 0.33
        if ratio_btn == 'Custom':
            Dataframe.ratio = st.sidebar.number_input('ratio', min_value=0.01, max_value=0.99)
        
        
class Classifier:
    """
    A class that handles classification tasks. It includes methods to initiate classification on static datasets or live packet data,
    and interactively configure and use various machine learning models.
    """
    
    #Defining dynamic parameter generation here
    def add_parameters(clf_name):
        """
        Based on the selected classifier, dynamically generates a form in the sidebar to allow the user
        to configure the classifier's parameters.

        Args:
            clf_name (str): The name of the classifier selected by the user.

        Returns:
            dict: A dictionary containing the selected parameters and their values.
        """
        
        params = dict()
        if clf_name == 'SVM':
            with st.sidebar.form('SVM Form'):
                C = st.slider('C', 0.01, 10.0)
                params['C'] = C
                kernel = st.selectbox('Select kernel',('rbf', 'linear', 'poly', 'sigmoid', 'precomputed'))
                params['kernel'] = kernel
                degree = st.selectbox('Select Custom/Default degree',('Default', 'Custom'))
                if degree == 'Default':
                    params['degree'] = 3
                if degree == 'Custom':
                    degree = st.number_input('degree', min_value=1, max_value=99999999)
                    params['degree'] = degree
                probability = st.checkbox('Enable probability estimates (uses 5-fold cross-validation)')
                if probability:
                    params['probability'] = True
                else:
                    params['probability'] = False

                svm_submit = st.form_submit_button('Apply Selected Options')

        if clf_name == 'KNN':
            with st.sidebar.form('KNN Form'):
                K = st.slider('K', 1, 15)
                params['K'] = K
                algorithm = st.selectbox('Select algorithm',('auto', 'ball_tree', 'kd_tree', 'brute'))
                params['algorithm'] = algorithm
                p = st.selectbox('Select Custom/Default Power (p)',('Default', 'Custom'))
                if p == 'Default':
                    params['p'] = 2
                if p == 'Custom':
                    p = st.number_input('p', min_value=1, max_value=99999999)
                    params['p'] = p
                n_jobs = st.selectbox('Select Custom/Default n_jobs (Parallel Jobs)',('Default', 'Custom'))
                if n_jobs == 'Default':
                    params['n_jobs'] = None
                if n_jobs == 'Custom':
                    n_jobs = st.number_input('n_jobs', min_value=-1, max_value=99999999)
                    params['n_jobs'] = n_jobs

                knn_submit = st.form_submit_button('Apply Selected Options')
            

        if clf_name == 'Random Forest':
            with st.sidebar.form('RF Form'):
                max_depth = st.slider('max_depth', 2, 15)
                params['max_depth'] = max_depth
                n_estimators = st.slider('n_estimators', 1, 100)
                params['n_estimators'] = n_estimators
                min_samples_split = st.selectbox('Select Custom/Default min_samples_split',('Default', 'Custom'))
                if min_samples_split == 'Default':
                    params['min_samples_split'] = 2
                if min_samples_split == 'Custom':
                    min_samples_split = st.number_input('min_samples_split', min_value=2, max_value=99999999)
                    params['min_samples_split'] = min_samples_split
                n_jobs = st.selectbox('Select Custom/Default n_jobs (Parallel Jobs)',('Default', 'Custom'))
                if n_jobs == 'Default':
                    params['n_jobs'] = None
                if n_jobs == 'Custom':
                    n_jobs = st.number_input('n_jobs', min_value=-1, max_value=99999999)
                    params['n_jobs'] = n_jobs
                criterion = st.selectbox('Select criterion',('gini', 'entropy'))
                params['criterion'] = criterion
                
                rf_submit = st.form_submit_button('Apply Selected Options')

        if clf_name == 'Decision Tree':
            with st.sidebar.form('DT Form'):
                criterion = st.selectbox('Select criterion',('gini', 'entropy'))
                params['criterion'] = criterion
                splitter = st.selectbox('Select splitter',('best', 'random'))
                params['splitter'] = splitter
                depth_type = st.selectbox('Select Custom/Default Tree Depth',('Default', 'Custom'))
                if depth_type == 'Default':
                    params['max_depth'] = None
                if depth_type == 'Custom':
                    max_depth = st.slider('max_depth', 2, 15)
                    params['max_depth'] = max_depth
                min_samples_split = st.selectbox('Select Custom/Default min_samples_split',('Default', 'Custom'))
                if min_samples_split == 'Default':
                    params['min_samples_split'] = 2
                if min_samples_split == 'Custom':
                    min_samples_split = st.number_input('min_samples_split', min_value=2, max_value=99999999)
                    params['min_samples_split'] = min_samples_split
                min_samples_leaf = st.selectbox('Select Custom/Default min_samples_leaf',('Default', 'Custom'))
                if min_samples_leaf == 'Default':
                    params['min_samples_leaf'] = 1
                if min_samples_leaf == 'Custom':
                    min_samples_leaf = st.number_input('min_samples_leaf', min_value=1, max_value=99999999)
                    params['min_samples_leaf'] = min_samples_leaf
                
                dt_submit = st.form_submit_button('Apply Selected Options')
            

        if clf_name == 'Logistic Regression':
            with st.sidebar.form('LR Form'):
                max_iter = st.selectbox('Select Custom/Default Iterations Number',('Default', 'Custom'))
                if max_iter == 'Default':
                    params['max_iter'] = 100
                if max_iter == 'Custom':
                    max_iter = st.number_input('max_iter', min_value=1, max_value=999999999999999)
                    params['max_iter'] = max_iter
                solver = st.selectbox('Select Solver',('lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'))
                params['solver'] = solver
                penalty = st.selectbox('Select penalty',('l2', 'l1', 'elasticnet', 'none'))
                params['penalty'] = penalty
                dual = st.checkbox('Enable Dual formulation')
                if dual:
                    params['dual'] = True
                else:
                    params['dual'] = False
                n_jobs = st.selectbox('Select Custom/Default n_jobs (Parallel Jobs)',('Default', 'Custom'))
                if n_jobs == 'Default':
                    params['n_jobs'] = None
                if n_jobs == 'Custom':
                    n_jobs = st.number_input('n_jobs', min_value=-1, max_value=99999999)
                    params['n_jobs'] = n_jobs

                LR_submit = st.form_submit_button('Apply Selected Options')

        if clf_name == 'Gradient Boosting Classifier':
            with st.sidebar.form('GBC Form'):
                n_estimators = st.selectbox('Select Custom/Default No. of Estimators',('Default', 'Custom'))
                if n_estimators == 'Default':
                    params['n_estimators'] = 100
                if n_estimators == 'Custom':
                    n_estimators = st.number_input('n_estimators', min_value=1, max_value=999999999999999)
                    params['n_estimators'] = n_estimators
                loss = st.selectbox('Loss Function',('deviance', 'exponential'))
                params['loss'] = loss
                max_depth = st.selectbox('Select Custom/Default max_depth',('Default', 'Custom'))
                if max_depth == 'Default':
                    params['max_depth'] = 3
                if max_depth == 'Custom':
                    max_depth = st.number_input('max_depth', min_value=1, max_value=999999999999)
                    params['max_depth'] = max_depth

                GBC_submit = st.form_submit_button('Apply Selected Options')

        if clf_name == 'LSTM':
            st.write(" ")

        if clf_name == 'Neural Networks':
            layer_no = st.sidebar.number_input("number of hidden layers", 1, 5, 1)
            with st.sidebar.form('NN Form'):
                layers = []
                for i in range(layer_no):
                    n_neurons = st.number_input(
                        f"Number of neurons at layer {i+1}", 2, 200, 100, 25
                    )
                    layers.append(n_neurons)
                layers = tuple(layers)
                params = {"hidden_layer_sizes": layers}

                NN_submit = st.form_submit_button('Apply Selected Options')

        return params


    #Defining prediction & accuracy metrics function here
    def get_prediction():
            """
            Initiates the classification process on the uploaded dataset. It splits the data into training and testing sets,
            trains the selected classifier, makes predictions, and computes accuracy and other metrics. It also provides the option to generate a report.
            """
        
            if st.button('Classify'):
                if uploaded_file is None:
                    st.error("Please upload a packet dataset before performing a classification task")
                    return
                st.write('Train to Test Ratio = ', Dataframe.ratio)
                

                #Splitting training and testing dataframes here
                Dataframe.X_train, Dataframe.X_test, Dataframe.y_train, Dataframe.y_test = train_test_split(Dataframe.X, Dataframe.y, test_size=Dataframe.ratio, random_state=1234)
                
                classifier_factory = ClassifierFactory()
                clf = classifier_factory.build_classifier(classifier_name, params)
                st.write(f'Classifier = {classifier_name}')
                with st.spinner('Classification in progress...'):

                    #Start classifier fitting and evaluation
                    start_time = time.time()
                    clf.fit(Dataframe.X_train, Dataframe.y_train)
                    end_time = time.time()
                    st.write("Training time: ",end_time-start_time, "seconds")

                    start_time = time.time()
                    Dataframe.y_pred = clf.predict(Dataframe.X_test)
                    end_time = time.time()
                    st.write("Prediction time: ",end_time-start_time, "seconds")

                    acc = accuracy_score(Dataframe.y_test, Dataframe.y_pred)
                    st.write('Accuracy =', acc)
                    metrics = sklearn.metrics.classification_report(Dataframe.y_test, Dataframe.y_pred)
                    st.text(metrics)
                    st.write("Train score is:", clf.score(Dataframe.X_train, Dataframe.y_train))
                    st.write("Test score is:",clf.score(Dataframe.X_test, Dataframe.y_test))
                    
                    if report_btn:
                        report = sklearn.metrics.classification_report(Dataframe.y_test, Dataframe.y_pred, output_dict=True)
                        Output.generate_report(report)

                    st.success('Done!')
                
            else: 
                st.write('Click the button to classify')


    def get_live_packet_prediction():
            """
            Initiates the classification process on live packet data. It preprocesses the live packet data,
            fits the classifier to the training data, and then makes predictions on the live data.
            """
        
            if st.button('Classify'):
                if uploaded_file is None:
                    st.error("Please upload a packet dataset before performing a classification task")
                    return
                st.write('Train to Test Ratio = ', Dataframe.ratio)
                
                #Splitting training and testing dataframes here
                Dataframe.X_train, Dataframe.X_test, Dataframe.y_train, Dataframe.y_test = train_test_split(Dataframe.X, Dataframe.y, test_size=Dataframe.ratio, random_state=1234)
                classifier_factory = ClassifierFactory()
                clf = classifier_factory.build_classifier(classifier_name, params)

        
                path2 = "packet_capture.csv"
                Dataframe.X2_test = pd.read_csv(path2)
                Dataframe.X2_test = Dataframe.X2_test.dropna(axis=0)
                
                #Keep only common columns between training data and live packet capture dataframes
                common_cols = [col for col in set(Dataframe.X_test.columns).intersection(Dataframe.X2_test.columns)]
                Dataframe.X_test = Dataframe.X_test[common_cols]
                Dataframe.X_train = Dataframe.X_train[common_cols]
                if len(common_cols)<1:
                    st.error("Please use a compatible model training dataset with the live packet capture feature scheme.")
                    return
                
                #Label encode the imported live packet capture dataframe
                num_cols2 = Dataframe.X2_test._get_numeric_data().columns
                cate_cols2 = list(set(Dataframe.X2_test.columns)-set(num_cols2))
                encoder=LabelEncoder()
                for item in cate_cols2:
                    Dataframe.X2_test[item] = encoder.fit_transform(Dataframe.X2_test[item])
                
                
                #Start classifier fitting and evaluation
                clf.fit(Dataframe.X_train, Dataframe.y_train)
                Dataframe.y_pred = clf.predict(Dataframe.X_test)
                acc = accuracy_score(Dataframe.y_test, Dataframe.y_pred)
                st.write(f'Classifier = {classifier_name}')
                st.write('Accuracy =', acc)
                metrics = sklearn.metrics.classification_report(Dataframe.y_test, Dataframe.y_pred)
                st.text(metrics)
                

                #Predict and show attacks in live packet capture dataframe
                Dataframe.y2_pred = clf.predict(Dataframe.X2_test)
                sample_df2 = pd.DataFrame(Dataframe.y2_pred, columns = ['Label']) #allow for a dynamic target label when you allow live packet capture of other datasets
                sample_df2 = sample_df2.sample(min(200, len(sample_df2)))
                st.write(sample_df2)
                st.area_chart(Dataframe.y2_pred)
                unique_elements, counts_elements = np.unique(Dataframe.y2_pred, return_counts=True)
                st.write("Frequency of unique values of the said array:")
                st.text(np.asarray((unique_elements, counts_elements)))
                
            else: 
                st.write('Click the button to classify')



#Instantiating the classifier selected in the sidebar --> Applying Python Factory Pattern Here (Callable Factory)
class ClassifierFactory(object):
    """
    Factory class for creating classifier instances based on the user's choice. It supports various classifiers and configures them with custom parameters.
    """
    def build_classifier(self, clf_name, params): #Foremely get_classifier
        """
        Builds and returns a classifier instance based on the provided name and parameters.

        Args:
            clf_name (str): The name of the classifier.
            params (dict): A dictionary of parameters for the classifier.

        Returns:
            A classifier instance configured with the specified parameters.
        """
        
        clf = None
        if clf_name == 'SVM':
            from sklearn.svm import SVC
            clf = SVC(C=params['C'], kernel=params['kernel'], degree=params['degree'])
        if clf_name == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(n_neighbors=params['K'], algorithm=params['algorithm'], p=params['p'], n_jobs=params['n_jobs'])
        if clf_name == 'Naive Bayes':
            from sklearn.naive_bayes import GaussianNB
            clf = GaussianNB()
        if clf_name == 'Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], random_state=1234, min_samples_split=params['min_samples_split'], n_jobs=params['n_jobs'], criterion=params['criterion'])
        if clf_name == 'Decision Tree':
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier(criterion=params['criterion'], splitter=params['splitter'], max_depth = params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'])
        if clf_name == 'Logistic Regression':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=params['max_iter'], solver=params['solver'], penalty=params['penalty'], n_jobs=params['n_jobs'])
        if clf_name == 'Gradient Boosting Classifier':
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], loss=params['loss'], max_depth=params['max_depth'])
        if clf_name == 'LSTM':
            from keras.wrappers.scikit_learn import KerasClassifier
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.layers import LSTM
            def lstm():
                model = Sequential()
                model.add(Dense(40,input_dim =41,activation = 'relu',kernel_initializer='random_uniform'))
                model.add(Dense(1,activation='sigmoid',kernel_initializer='random_uniform'))
                
                model.compile(loss ='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
                return model
            clf = KerasClassifier(build_fn=lstm,epochs=3,batch_size=64)
        if clf_name == 'Neural Networks':
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(**params)
        return clf
        


class Output:
    """
    Class to handle output operations like generating and displaying reports of classification metrics.
    """
    def generate_report(report):
        """
        Generates and saves a classification report to a file.

        Args:
            report (dict): A dictionary containing classification metrics.
        """
        
        stdf = pd.DataFrame(report).transpose()
        st.dataframe(stdf)
        stdf.to_csv (f'MetricsReports\{classifier_name}.csv', index = True, header=True)

    def show_metrics_reports():
        """
        Displays all generated metrics reports from the specified directory.
        """
        
        st.header("Metrics Reports: ")
        directory="MetricsReports"
        for filename in os.listdir(directory):
            if filename.endswith(".csv"): 
                st.subheader(filename[:-4])
                metric_csv = pd.read_csv(os.path.join(directory, filename))
                st.write(metric_csv)
                continue
            else:
                continue


uploaded_file = None

prediction_type = st.selectbox('Select Imported Dataset or Live Packet Data Prediction', ['Imported Dataset','Live Packet Data'])

if prediction_type == 'Connect to AWS S3 Database':
    uploaded_file = DBConnection.fetchDB()

else:
    uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    DataInput.read_data()

#Populating classification sidebar here
st.sidebar.header("Classification")
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Naive Bayes', 'KNN', 'SVM', 'Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting Classifier', 'LSTM', 'Neural Networks')
)
params = Classifier.add_parameters(classifier_name)


if prediction_type == 'Live Packet Data':
    PacketCapture.populate_live_capture()


report_btn = st.checkbox("Add classification task to Metrics Report")

if(prediction_type == 'Imported Dataset' or prediction_type == 'Connect to AWS S3 Database'):
    Classifier.get_prediction()
if(prediction_type == 'Live Packet Data'):
    Classifier.get_live_packet_prediction()


st.sidebar.subheader("Metrics Reports")
metrics_reports_btn = st.sidebar.button("Show Metrics Reports")
if metrics_reports_btn:
    Output.show_metrics_reports()

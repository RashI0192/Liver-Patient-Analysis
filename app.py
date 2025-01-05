import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


    
def readcsv():
    data=pd.read_csv("liver_data.csv")
    
    data.dropna(inplace=True)
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    
    return data


def sidebar():
    st.sidebar.header("Info")
    data=readcsv()
    # List of labels and column names
    slider_labels = [
        ('Age', 'Age'),
        ('Gender', 'Gender'),
        ('Total Bilirubin', 'Total_Bilirubin'),
        ('Direct Bilirubin', 'Direct_Bilirubin'),
        ('Alkaline Phosphotase', 'Alkaline_Phosphotase'),
        ('Alamine Aminotransferase', 'Alamine_Aminotransferase'),
        ('Aspartate Aminotransferase', 'Aspartate_Aminotransferase'),
        ('Total Proteins', 'Total_Protiens'),
        ('Albumin', 'Albumin'),
        ('Albumin and Globulin Ratio', 'Albumin_and_Globulin_Ratio')
        
    ]
    input_dict={}
    for  label,key in slider_labels:
        input_dict[key]=st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),#min max both need to be float
            value=float(data[key].mean())
        )
    return input_dict
        
#Builing  a Radar Chart
def radar_chart(input_data):#from plotly
    input_data=scale(input_data)

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['Age'], input_data['Gender'], input_data['Total_Bilirubin'],#Radial value distance from centre
                input_data['Direct_Bilirubin'], input_data['Alkaline_Phosphotase'], input_data['Alamine_Aminotransferase'],
                input_data['Total_Protiens'], input_data['Albumin'], input_data['Albumin_and_Globulin_Ratio']#r input value
                ],
            theta=['Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin', 'Alkaline Phosphotase', 'Aspartate Aminotransferase', 
                   'Total Proteins', 'Albumin',
                   'Albumin and Globulin Ratio'],#name
            fill='toself',
            name='Information'
        )
    )

    

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]#scaling to help with display
            )
        ),
        showlegend=True,
        autosize=True
    )

    return fig

    
def scale(input_dict):
    # Define a Function to Scale the Values based on the Min and Max of the Predictor in the Training Data
    data = readcsv()
    X = data.drop(['Dataset'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()#or can use scaler from skit-learn
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def add_predictions(input_data):
    model = pickle.load(open("model.pkl", "rb"))  # read in binary
    scaler = pickle.load(open("scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values()))
    input_array = input_array.reshape(1, -1)  # Reshape to 2D array
    
    input_array_scaled = scaler.transform(input_array)  # Scale the input array
    
    st.write("Input array:", input_array)
    st.write("Scaled input array:", input_array_scaled)

    prediction = model.predict(input_array_scaled)
    st.write("Prediction:", prediction)
    
    probability = model.predict_proba(input_array_scaled)
    st.write("Probability of being benign:", probability[0][0])
    st.write("Probability of being malignant:", probability[0][1])

    st.write("App only made to help, not replace")
    





def main():
    st.set_page_config(
        page_title="Liver Disease Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    with st.container():
        st.title("Liver Disease Detector")
        st.write("Please connect to your labatory or put in the values by hand :)")
    col1,col2=st.columns([4,1])#first one 4 times longer than second
    input_data=sidebar()
    with col1:
        radarchart=radar_chart(input_data)
        st.plotly_chart(radarchart)
    with col2:
        add_predictions(input_data)
    

if __name__=='__main__':
    main()
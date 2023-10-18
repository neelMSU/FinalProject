import streamlit as st
import time
from st_pages import Page, show_pages, add_page_title
import plotly.graph_objects as go
import pandas as pd

with st.spinner("Please wait while detail analysis is loading."):
    time.sleep(5)
    df=pd.read_csv('MidtermProject/diabetes.csv')
    df['Glucose']=df['Glucose'].replace(0, df['Glucose'].mean())
    df['BloodPressure']=df['BloodPressure'].replace(0, df['BloodPressure'].mean())
    df['SkinThickness']=df['SkinThickness'].replace(0, df['SkinThickness'].mean())
    df['Insulin']=df['Insulin'].replace(0, df['Insulin'].mean())
    df['BMI']=df['BMI'].replace(0, df['BMI'].mean())
    df['Age']=df['Age'].replace(0, df['Age'].mean())

    # Optional -- adds the title and icon to the current page
    add_page_title()
    oc=[]
    for i in df['Outcome']:
        if i==0:
            oc.append('Person is with no Diabetes')
        else:
            oc.append("Person is with Diabetes")

    # Specify what pages should be shown in the sidebar, and what their titles and icons
    # should be
    show_pages(
        [
            Page("MidtermProject/MidtermProject.py", "Introduction to Dataset", "📊"),
            Page("MidtermProject/MidtermProject2.py", "Detailed Analysis", "📈"),
            Page("MidtermProject/MidtermProject3.py", "Deep Features Study", "🧐"),
            Page("MidtermProject/MidtermProject4.py", "Our Conclusion", "🎯")
        ]
)
    streamlit_style = """
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

                html, body, [class*="css"]  {
                font-family: 'Roboto', sans-serif;
                }
                </style>
                """
    st.markdown(streamlit_style, unsafe_allow_html=True)
    st.header("In introduction we saw that Glucose, BMI, Pregnancy and Age are four important parameters doctors should look upon.")
    st.header("Let's go ahead and do some detailed analysis on these parameters.")
    st.header("Using below plot, we can generate 3D plot for our four important features from our dataset.")
    lst=['Glucose', 'BMI', 'Pregnancies','Age']
    co1,co2,co3=st.columns(3)
    with co1:
        option1 = st.selectbox(
            'For X',
            lst)
    with co2:
        option2 = st.selectbox(
            'For Y',
            lst)
    with co3:
        option3 = st.selectbox(
            'For Z',
            lst)
    op=[option1,option2,option3]
    fig = go.Figure(data =[go.Scatter3d(x = df[option1],
                                    y = df[option2],
                                    z = df[option3],
                                    text=oc,
                                    mode ='markers', 
                                    marker = dict(
                                        size = 10,
                                        color = df['Outcome'],
                                        colorscale ='Viridis',
                                        opacity = 0.8,
                                    )
        )])
    fig.update_layout(width=800, height=800,
                  scene = dict(xaxis=dict(title=op[0], titlefont_color='black'),
                               yaxis=dict(title=op[1], titlefont_color='black'),
                               zaxis=dict(title=op[2], titlefont_color='black')
                           ))
    st.plotly_chart(fig,use_container_width=True)
    st.write("If in case you want to generate plot for all the features from dataset, we have a small plot for that but, we won't be focusing on all the features.")

    lst=['Pregnancies','Glucose','BloodPressure','SkinThickness',"Insulin",'BMI']
    co1,co2,co3=st.columns(3)
    with co1:
        option1 = st.selectbox(
            'For X',
            lst)
    with co2:
        option2 = st.selectbox(
            'For Y',
            lst)
    with co3:
        option3 = st.selectbox(
            'For Z',
            lst)
    op=[option1,option2,option3]
    fig = go.Figure(data =[go.Scatter3d(x = df[option1],
                                    y = df[option2],
                                    z = df[option3],
                                    text=oc,
                                    mode ='markers', 
                                    marker = dict(
                                        size = 10,
                                        color = df['Outcome'],
                                        colorscale ='Viridis',
                                        opacity = 0.8,
                                    )
        )])
    fig.update_layout(width=400, height=400,
                  scene = dict(xaxis=dict(title=op[0], titlefont_color='black'),
                               yaxis=dict(title=op[1], titlefont_color='black'),
                               zaxis=dict(title=op[2], titlefont_color='black')
                           ))
    st.plotly_chart(fig,use_container_width=True)
    st.header("Further ahead, let's try to go in deep with our 4 important features in Deep Features Study.")
    

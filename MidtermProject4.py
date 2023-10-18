import streamlit as st
from st_pages import Page, show_pages, add_page_title
import time
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
streamlit_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

            html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
            }
            </style>
            """
st.markdown(streamlit_style, unsafe_allow_html=True)

with st.spinner("Let's have a look at our conclusion."):
    time.sleep(5)

# Optional -- adds the title and icon to the current page
    add_page_title()

    # Specify what pages should be shown in the sidebar, and what their titles and icons
    # should be
    show_pages(
        [
            Page("MidtermProject.py", "Introduction to Dataset", "📊"),
            Page("MidtermProject2.py", "Detailed Analysis", "📈"),
            Page("MidtermProject3.py", "Deep Features Study", "🧐"),
            Page("MidtermProject4.py", "Our Conclusion", "🎯")
        ]
)

st.header("From our study we realised that BMI, Age, Glucose and number of Pregnancies are the factors doctors should look upon in order to make sure person is not having conditions to have Diabetes.")

st.header("Doctors should make sure that their patients glucose is not above 120, BMI is below 20, if person had they should have maximum three pregnancies, Age is below 23. Anything above these limits and alarm should sound in mind which will let doctor to tell patient of their health care. There parameters that could be controlled here are Glucose and BMI. These features of body should be regularly monitered by doctor of every patient facing danger of Diabetes to make sure he dosen't catches diabetes.")

st.header("A 3D plot just to generate all the features if further interest.")
df=pd.read_csv('/Users/neeljoshi/Desktop/CSE830/diabetes.csv')
df['Glucose']=df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure']=df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness']=df['SkinThickness'].replace(0, df['SkinThickness'].mean())
df['Insulin']=df['Insulin'].replace(0, df['Insulin'].mean())
df['BMI']=df['BMI'].replace(0, df['BMI'].mean())
df['Age']=df['Age'].replace(0, df['Age'].mean())
oc=[]
for i in df['Outcome']:
    if i==0:
        oc.append('Person is with no Diabetes')
    else:
        oc.append("Person is with Diabetes")

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
fig.update_layout(width=800, height=800,
                  scene = dict(xaxis=dict(title=op[0], titlefont_color='black'),
                               yaxis=dict(title=op[1], titlefont_color='black'),
                               zaxis=dict(title=op[2], titlefont_color='black')
                           ))
st.plotly_chart(fig,use_container_width=True)

image=Image.open('/Users/neeljoshi/Downloads/b38b2f0fc571a7d1d581e03f11d05619.png')
st.image(image)

st.header("Any Questions??")
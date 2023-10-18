import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time
import streamlit as st
from st_pages import Page, show_pages, add_page_title
with st.spinner("Welcome to my Diabetes analysis web app. Please wait while it's loading."):
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

    ##IN MY CODE I HAVE USED DIFFERENT FONT USING FOLLOWING####
    streamlit_style = """
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

                html, body, [class*="css"]  {
                font-family: 'Roboto', sans-serif;
                }
                </style>
                """
    st.markdown(streamlit_style, unsafe_allow_html=True)

    df=pd.read_csv('diabetes.csv')
    ## WHAT I REALIZED IN MY DATASET IS MISSING VALUES ARE IN THE FORM OF 0 INSTEAD OF NAN
    ##WHAT I'M GONNA DO IS REPLACE THOSE 0 WITH MEAN OF THE RESPECTIVE COLUMN
    df['Glucose']=df['Glucose'].replace(0, df['Glucose'].mean())
    df['BloodPressure']=df['BloodPressure'].replace(0, df['BloodPressure'].mean())
    df['SkinThickness']=df['SkinThickness'].replace(0, df['SkinThickness'].mean())
    df['Insulin']=df['Insulin'].replace(0, df['Insulin'].mean())
    df['BMI']=df['BMI'].replace(0, df['BMI'].mean())

    st.header("""
    For this project I'm going to use the Diabetes Dataset. What I'm trying to find out is most obvious health condition to be focused on inorder to prevent diabetes.
    """)
    st.dataframe(df.head())
    st.header("Diabetes is one of the most dangerious diesease which does effect person's daily lifestyle. Dealing with such disease should be top priority.")


    st.header("With the help of my dataset I will try to find out most probable cause for people with diabetes as well as for those with no diabetes.")

    colors = ['gold', 'mediumturquoise']
    labels = ['No Diabetes','With Diabetes']
    values = df['Outcome'].value_counts()/df['Outcome'].shape[0]

    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                    marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig.update_layout(
        title_text="Outcome")
    st.plotly_chart(fig)

    st.header("With our plot above we can see that, even though there are 65.1% people with no Diabetes, there's still 34.9% people with this Disease.")

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(df.corr(), ax=ax)
    st.write(fig)

    st.header("From the above figure we can see that correlation between Outcome and Glucose is high.")
    st.header("Thus Glucose can be considered as one of the most important factor to look upon. We can also consider BMI, Pregnancy and Age for our detailed analysis ahead.")
    st.header("Doctors should always look upon their patient's Glucose, BMI, Pregnancy and Age while determining their threat to get diabetes.")
    st.header("With these features identified, further let's focus on these features specifically in Detailed Analysys.")

import streamlit as st
import time
from st_pages import Page, show_pages, add_page_title
import pandas as pd
import altair as alt

with st.spinner("Let's do some detailed study on our four features!!"):
    time.sleep(5)
    df=pd.read_csv('diabetes.csv')
    df['Glucose']=df['Glucose'].replace(0, df['Glucose'].mean())
    df['BloodPressure']=df['BloodPressure'].replace(0, df['BloodPressure'].mean())
    df['SkinThickness']=df['SkinThickness'].replace(0, df['SkinThickness'].mean())
    df['Insulin']=df['Insulin'].replace(0, df['Insulin'].mean())
    df['BMI']=df['BMI'].replace(0, df['BMI'].mean())
    df['Age']=df['Age'].replace(0, df['Age'].mean())
    # df['Outcome'].replace([1,0],
    #                 ['Person is with Diabetes','Person is with no Diabetes'], inplace=True)

    add_page_title()
    show_pages(
        [
            Page("MidtermProject.py", "Introduction to Dataset", "📊"),
            Page("MidtermProject2.py", "Detailed Analysis", "📈"),
            Page("MidtermProject3.py", "Deep Features Study", "🧐"),
            Page("MidtermProject4.py", "Our Conclusion", "🎯")
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
st.header("Let's try to plot some histograms for our 4 important features we consider.")
lst=['Glucose', 'BMI', 'Pregnancies','Age']
option = st.selectbox(
            'Select Your feature to study.',
            lst)
hist = alt.Chart(df).mark_bar().encode(x = alt.X(option, 
                                                bin = alt.BinParams(maxbins = 30)), 
                                              y = 'count()',color = 'Outcome').properties(width=680,height=500) 
st.altair_chart(hist)
tex=['We can see that most people with Diabetes have Glucose range of 120-130. We can infer that Glucose beyond 120 is most dangerious. Also most people without diabetes have gluose range of 100-110. So we can infer that inorder to remain safe the maximum Glucose value is 100.',
"For BMI the care should be taken that the value is within 20-22. Because that's the value where people are not having Diabetes.",
"Anyone with 3-4 pregnancies should take care of their health as there are more people with diabetes. So people with more than 3 pregnancies should take care of their health most.",
"Age is one of the most important and crucial factor to be considered while analyzing how someone is prone to get Diabetes. Turns out that people with age 23 and beyond as abovious should take super care of their health if they don't want to get affected by Diabetes. We can see most people with Diabetes are with age range of 28-30. So let's say average age of 25 would be a best point to be start taking care."]
if option==lst[0]:
    st.header(tex[0])
elif option==lst[1]:
    st.header(tex[1])
elif option==lst[2]:
    st.header(tex[2])
else:
    st.header(tex[3])

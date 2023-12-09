import streamlit as st
import time
from st_pages import Page, show_pages, add_page_title
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

with st.spinner("Please wait while detail analysis is loading."):
    time.sleep(.4)

    # Optional -- adds the title and icon to the current page
    add_page_title()
    oc=[]

    show_pages(
        [
            Page("MidtermProject.py", "Introduction to Dataset", "📊"),
            Page("MidtermProject2.py", "Detailed Analysis", "📈"),
            Page("MidtermProject3.py", "Deep Features Study", "🧐"),
            Page("Proj.py", "Bringing in the Power ML", "🧠"),
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
    st.header("Now you just saw that Glucose, BMI, Pregnancy and Age are four important parameters doctors should look upon.")
    st.header("But what if you were able to predict whether someone has a Diabetes or not based on above parameters??")
    st.header("Well you can do that, by bringin in the power of ML(Machine Learning)!!!!")
    st.header("You can bring in the power 💪 of Random Forest 🌳 Algorithm. Random Forest helps to implement ML in computer with the help of Python and since you have ML in picture, it means you are bringing in human brain power!!!")


    st.header("After doing training and testing, you will get a solid 100 percent of accuracy. This means the model for classification created, is predicting correctly whether someone has Diabetes or not based on selected inputs you provide.")
    image=Image.open('img2.png')
    st.image(image)

    st.header("Below you will also be able to see a confusion matrix. A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. It provides a summary of the model's predictions and how well they align with the actual classes.")
    image=Image.open('img1.png')
    st.image(image)
    with st.expander("Wanna learn what Confusion Matrix is!!!"):
        st.header("True Positive: The cases where the model predicted the positive class, and the actual class is also positive.")
        st.header("True Negative: The cases where the model predicted the negative class, and the actual class is also negative.")
        st.header("False Positive: The cases where the model predicted the positive class, but the actual class is negative.")
        st.header("False Negative: The cases where the model predicted the negative class, but the actual class is positive")
    st.header('Below 👇 you will see the ML model implemented to check whether someone has Diabetes or not')
    st.header('Play around it 😄')
    
    with st.form("my_form"):
        st.write("ML model")
        pregnancy=st.number_input("Enter Number of pregnancies")
        glucose=st.number_input("Enter Glucose level")
        bmi=st.number_input("Enter BMI level")
        age = st.slider("Enter Age")





        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report

        # Load the CSV dataset
        df = pd.read_csv('diabetes.csv')  # Replace with your actual dataset path
        df['Glucose']=df['Glucose'].replace(0, df['Glucose'].mean())
        df['BloodPressure']=df['BloodPressure'].replace(0, df['BloodPressure'].mean())
        df['SkinThickness']=df['SkinThickness'].replace(0, df['SkinThickness'].mean())
        df['Insulin']=df['Insulin'].replace(0, df['Insulin'].mean())
        df['BMI']=df['BMI'].replace(0, df['BMI'].mean())
        df['Age']=df['Age'].replace(0, df['Age'].mean())
        df = df.drop(['BloodPressure','SkinThickness','Insulin','DiabetesPedigreeFunction'], axis=1)
        X = df.iloc[:, :-1]  # Features
        y = df.iloc[:, -1] 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        classifier.fit(X_train, y_train)

        print(X_test)
        data=[[pregnancy,glucose,bmi,age]]
        import pandas as pd
        predictions = classifier.predict(pd.DataFrame(data))

        submitted = st.form_submit_button("Submits")
        if submitted:
            if predictions==1:
                st.warning('Warning!!! You have Diabetes',icon="⚠️")
            else:
                st.success('Good!!! No Diabetes', icon="✅")




# import streamlit as st
# import time
# from st_pages import Page, show_pages, add_page_title
# import plotly.graph_objects as go
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import sklearn.model_selection
# from sklearn.model_selection import train_test_split

# with st.spinner("Please wait while detail analysis is loading."):
#     time.sleep(.5)
#     df=pd.read_csv('/Users/neeljoshi/Downloads/CSE830/Project/diabetes.csv')
#     df['Glucose']=df['Glucose'].replace(0, df['Glucose'].mean())
#     df['BloodPressure']=df['BloodPressure'].replace(0, df['BloodPressure'].mean())
#     df['SkinThickness']=df['SkinThickness'].replace(0, df['SkinThickness'].mean())
#     df['Insulin']=df['Insulin'].replace(0, df['Insulin'].mean())
#     df['BMI']=df['BMI'].replace(0, df['BMI'].mean())
#     df['Age']=df['Age'].replace(0, df['Age'].mean())

#     # Optional -- adds the title and icon to the current page
#     add_page_title()
#     oc=[]
#     for i in df['Outcome']:
#         if i==0:
#             oc.append('Person is with no Diabetes')
#         else:
#             oc.append("Person is with Diabetes")

#     # Specify what pages should be shown in the sidebar, and what their titles and icons
#     # should be
#     show_pages(
#         [
#             Page("/Users/neeljoshi/Downloads/CSE830/Project/MidtermProject.py", "Introduction to Dataset", "📊"),
#             Page("/Users/neeljoshi/Downloads/CSE830/Project/MidtermProject2.py", "Detailed Analysis", "📈"),
#             Page("/Users/neeljoshi/Downloads/CSE830/Project/MidtermProject3.py", "Deep Features Study", "🧐"),
#             Page("/Users/neeljoshi/Downloads/CSE830/Project/MidtermProject5.py", "Bringing in the Power of Neurons", "🧠"),
#             Page("/Users/neeljoshi/Downloads/CSE830/Project/MidtermProject4.py", "Our Conclusion", "🎯")
#         ]
#     )
#     streamlit_style = """
#                 <style>
#                 @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

#                 html, body, [class*="css"]  {
#                 font-family: 'Roboto', sans-serif;
#                 }
#                 </style>
#                 """
#     st.markdown(streamlit_style, unsafe_allow_html=True)
#     st.header("Now you just saw that Glucose, BMI, Pregnancy and Age are four important parameters doctors should look upon.")
#     st.header("But what if you were able to predict whether someone has a Diabetes or not based on above parameters??")
#     st.header("Well you can do that, by bringin in the power of Neurons!!")
#     st.header("You can bring in the power of Neurons by using Tensorflow. Tensorflow helps to implement neurons in computer with the help of Python and since you have neurons in picture, it means you are bringing in human brain power in big picture!!!")
#     df=pd.read_csv('diabetes.csv')
#     df['Glucose']=df['Glucose'].replace(0, df['Glucose'].mean())
#     df['BloodPressure']=df['BloodPressure'].replace(0, df['BloodPressure'].mean())
#     df['SkinThickness']=df['SkinThickness'].replace(0, df['SkinThickness'].mean())
#     df['Insulin']=df['Insulin'].replace(0, df['Insulin'].mean())
#     df['BMI']=df['BMI'].replace(0, df['BMI'].mean())
#     df['Age']=df['Age'].replace(0, df['Age'].mean())


#     X = df.iloc[:,:-1].values
#     Y = df.iloc[:,-1].values
#     X = df.drop(['BloodPressure','SkinThickness','Insulin','DiabetesPedigreeFunction'], axis=1)
#     y = df['Outcome']
#     X_train, X_test, y_train, y_test = train_test_split(
#     X, y, 
#     test_size=0.2, random_state=42
#     )
#     X_train,y_train
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     X_train_scaled
#     tf.random.set_seed(42)
#     model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
#     model.compile(
#     loss=tf.keras.losses.binary_crossentropy,
#     optimizer=tf.keras.optimizers.Adam(lr=0.03),
#     metrics=[
#         tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#         tf.keras.metrics.Precision(name='precision'),
#         tf.keras.metrics.Recall(name='recall')
#     ]
#     )
#     history = model.fit(X_train_scaled, y_train, epochs=100)

    # import matplotlib.pyplot as plt
    # from matplotlib import rcParams
    # rcParams['figure.figsize'] = (18, 8)
    # rcParams['axes.spines.top'] = False
    # rcParams['axes.spines.right'] = False
    # st.pyplot(plt.plot(
    # np.arange(1, 101), 
    # history.history['loss'], label='Loss'
    # ))
    # st.pyplot(plt.plot(
    # np.arange(1, 101), 
    # history.history['accuracy'], label='Accuracy'
    # ))
    # st.pyplot(plt.plot(
    # np.arange(1, 101), 
    # history.history['precision'], label='Precision'
    # ))
    # st.pyplot(plt.plot(
    # np.arange(1, 101), 
    # history.history['recall'], label='Recall'
    # ))

    







    # lst=['Glucose', 'BMI', 'Pregnancies','Age']
    # co1,co2,co3=st.columns(3)
    # with co1:
    #     option1 = st.selectbox(
    #         'For X',
    #         lst)
    # with co2:
    #     option2 = st.selectbox(
    #         'For Y',
    #         lst)
    # with co3:
    #     option3 = st.selectbox(
    #         'For Z',
    #         lst)
    # op=[option1,option2,option3]
    # fig = go.Figure(data =[go.Scatter3d(x = df[option1],
    #                                 y = df[option2],
    #                                 z = df[option3],
    #                                 text=oc,
    #                                 mode ='markers', 
    #                                 marker = dict(
    #                                     size = 10,
    #                                     color = df['Outcome'],
    #                                     colorscale ='Viridis',
    #                                     opacity = 0.8,
    #                                 )
    #     )])
    # fig.update_layout(width=800, height=800,
    #               scene = dict(xaxis=dict(title=op[0], titlefont_color='black'),
    #                            yaxis=dict(title=op[1], titlefont_color='black'),
    #                            zaxis=dict(title=op[2], titlefont_color='black')
    #                        ))
    # st.plotly_chart(fig,use_container_width=True)
    # st.write("If in case you want to generate plot for all the features from dataset, we have a small plot for that but, we won't be focusing on all the features.")

    # lst=['Pregnancies','Glucose','BloodPressure','SkinThickness',"Insulin",'BMI']
    # co1,co2,co3=st.columns(3)
    # with co1:
    #     option1 = st.selectbox(
    #         'For X',
    #         lst)
    # with co2:
    #     option2 = st.selectbox(
    #         'For Y',
    #         lst)
    # with co3:
    #     option3 = st.selectbox(
    #         'For Z',
    #         lst)
    # op=[option1,option2,option3]
    # fig = go.Figure(data =[go.Scatter3d(x = df[option1],
    #                                 y = df[option2],
    #                                 z = df[option3],
    #                                 text=oc,
    #                                 mode ='markers', 
    #                                 marker = dict(
    #                                     size = 10,
    #                                     color = df['Outcome'],
    #                                     colorscale ='Viridis',
    #                                     opacity = 0.8,
    #                                 )
    #     )])
    # fig.update_layout(width=400, height=400,
    #               scene = dict(xaxis=dict(title=op[0], titlefont_color='black'),
    #                            yaxis=dict(title=op[1], titlefont_color='black'),
    #                            zaxis=dict(title=op[2], titlefont_color='black')
    #                        ))
    # st.plotly_chart(fig,use_container_width=True)
    # st.header("Further ahead, let's try to go in deep with our 4 important features in Deep Features Study.")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV dataset
dataset_path = '/Users/neeljoshi/Downloads/CSE830/Project/diabetes.csv'  # Replace with your actual dataset path
df = pd.read_csv(dataset_path)

# Assuming the last column is the target variable (class label)
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a classifier (Random Forest in this example)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the evaluation metrics
st.header(f"Accuracy: {accuracy:.2f}")
st.header("\nClassification Report:\n", classification_rep)


# st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache(allow_output_mutation=True)

# def load_model():
# 	model = keras.models.load_model('/Users/neeljoshi/Downloads/CSE830/Project/flower_model_trained.hdf5')
# 	return model

# def predict_class(model):

#     numpy_array = np.array([[ 1.8901091 , -0.69285836,  1.91225539,  0.44308379,  1.37208932]])
#     tensor1 = tf.convert_to_tensor(numpy_array)
#     predictions = model.predict(tensor1)
#     prediction = model.predict(predictions)

#     return prediction
# model = load_model()

# pred = predict_class(model)

st.write("Hello")

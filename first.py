import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt  

def main():
    st.title("Cancer Prediction App")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset")
        st.write(data)

       
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the KNN model
        n_neighbors = st.sidebar.slider("Select number of neighbors (k)", 1, 15, 5)
        classifiers = KNeighborsClassifier(n_neighbors=n_neighbors)
        classifiers.fit(X_train, y_train)

        # Prediction
        y_pred = classifiers.predict(X_test)

        # Metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)

        st.write(f"### Accuracy: {accuracy * 100:.2f}%")
        st.write(f"Recall: {recall * 100:.2f}%")
        st.write(f"Precision: {precision * 100:.2f}%")
        st.write(f"F1 Score: {f1 * 100:.2f}%")

        # Confusion Matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'], ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        st.pyplot(fig)

    


        # Sidebar
        st.sidebar.header("Make Predictions")
        input_data = {
            "Age": st.sidebar.number_input("Age", value=30),
            "Gender": st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female"),
            "BMI": st.sidebar.number_input("BMI", value=25.0),
            "Smoking": st.sidebar.selectbox("Smoking", options=[0, 1]),
            "GeneticRisk": st.sidebar.selectbox("Genetic Risk", options=[0, 1, 2]),
            "PhysicalActivity": st.sidebar.number_input("Physical Activity", value=5.0),
            "AlcoholIntake": st.sidebar.number_input("Alcohol Intake", value=2.0),
            "CancerHistory": st.sidebar.selectbox("Cancer History", options=[0, 1])
        }

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = classifiers.predict(scaled_input)
        result = "Positive" if prediction[0] == 1 else "Negative"

        st.write("### Prediction Result")
        st.write(f"The model predicts: **{result}**")

        # Bar Chart for Predictions
        st.write("### Prediction Distribution")
        prediction_counts = pd.Series(y_pred).value_counts()
        fig, ax = plt.subplots()
        prediction_counts.plot(kind='bar', color=['skyblue', 'orange'], ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'], rotation=0)
        ax.set_title('Prediction Distribution')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('Count')
        st.pyplot(fig)

if __name__ == "__main__":
    main()

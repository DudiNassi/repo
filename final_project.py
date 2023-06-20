import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import base64
import plotly.figure_factory as ff
import graphviz

# Load the dataset and train the decision tree model
diabetes = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
X = diabetes.drop(["Diabetes_binary"], axis=1)
y = diabetes["Diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=3, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)

# Generate the DOT data and create the graphviz visualization
features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
            'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
            ]
class_names = ['0', '1']

# Convert BMI feature to binary
diabetes['BMI'] = diabetes['BMI'].apply(lambda x: 1 if x >= 25 else 0)

# Split MentHlth and PhysHlth based on threshold
threshold = 14
diabetes['MentHlth'] = diabetes['MentHlth'].apply(lambda x: 1 if x >= threshold else 0)
diabetes['PhysHlth'] = diabetes['PhysHlth'].apply(lambda x: 1 if x >= threshold else 0)
diabetes['GenHlth'] = diabetes['GenHlth'].apply(lambda x: 1 if x >= 4 else 0)
diabetes['Income'] = diabetes['Income'].apply(lambda x: 1 if x >= 4 else 0)
diabetes['Education'] = diabetes['Education'].apply(lambda x: 1 if x >= 4 else 0)

# Calculate the percentage of diabetes and non-diabetes for each feature
features1 = list(set(diabetes.columns) - set(['Age', 'Diabetes_binary']))
diabetes_percentage = []
non_diabetes_percentage = []

for feature in features1:
    diabetes_counts = diabetes[diabetes[feature] == 1]['Diabetes_binary'].value_counts()
    total_counts = diabetes_counts.sum()

    diabetes_percentage.append(diabetes_counts[1] / total_counts * 100)
    non_diabetes_percentage.append(diabetes_counts[0] / total_counts * 100)

# Define the Streamlit app
def main():
    st.title("Diabetes Dataset Visualizations")

    # Bar Plot
    st.subheader("Diabetes vs. Non-Diabetes Percentage by Age Range")
    age_range = st.selectbox("Age Range", ['18-44', '45-69', '70-80 or older', 'All'])
    update_bar_plot(age_range)

    # Heatmap - Correlation Matrix    # Heatmap - Correlation Matrix
    st.subheader("Heatmap - Correlation Matrix")
    st.write("Correlation Matrix of Features")
    st.dataframe(diabetes[features].corr())

    # Decision Tree Visualization
    st.subheader("Decision Tree Visualization")
    dot_data = tree.export_graphviz(
        model,
        feature_names=features,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(graph)

def update_bar_plot(age_range):
    if age_range == 'All':
        diabetes_percentage_all = diabetes['Diabetes_binary'].value_counts(normalize=True) * 100
        non_diabetes_percentage_all = 100 - diabetes_percentage_all
        data = [
            ['Diabetes', diabetes_percentage_all[1]],
            ['Non-Diabetes', non_diabetes_percentage_all[1]]
        ]
    else:
        age_column = 'Age_' + age_range
        diabetes_counts = diabetes.groupby([age_column, 'Diabetes_binary']).size().unstack()
        total_counts = diabetes_counts.sum(axis=1)
        diabetes_percentage = diabetes_counts[1] / total_counts * 100
        non_diabetes_percentage = diabetes_counts[0] / total_counts * 100
        data = [
            ['Diabetes', diabetes_percentage[1]],
            ['Non-Diabetes', non_diabetes_percentage[1]]
        ]

    df = pd.DataFrame(data, columns=['Label', 'Percentage'])
    fig = ff.create_annotated_bar(
        df,
        x='Label',
        y='Percentage',
        colors=['#FF4F4F', '#73E68C'],
        annotation_text=df['Percentage'].astype(str) + '%',
        title="Diabetes vs. Non-Diabetes Percentage"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()


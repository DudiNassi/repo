import streamlit as st
import pandas as pd
import plotly.figure_factory as ff

# Load the dataset
diabetes = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
correlation_matrix = diabetes.corr().round(3)

def update_heatmap(age_range):
    if age_range == '1-13':
        min_age, max_age = 1, 13
    else:
        min_age, max_age = map(int, age_range.split('-'))
        st.text(min_age)
        st.text(max_age)
    filtered_diabetes = diabetes[(diabetes['Age'] >= min_age) & (diabetes['Age'] <= max_age)]
    correlation_matrix = filtered_diabetes.corr().round(3)

    heatmap = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index),
        colorscale='Viridis'
    )

    heatmap.update_layout(
        title='Correlation Matrix',
        xaxis_title='Variables',
        yaxis_title='Variables'
    )

    return heatmap

# Streamlit app
def main():
    st.title('Heatmap - Correlation Matrix by Age Range')

    # Dropdown for age range selection
    age_range = st.selectbox('Age Range', ['18-44', '45-69', '70-80 or older', 'All'], index=0)

    # Update the heatmap based on the selected age range
    heatmap = update_heatmap(age_range)

    # Display the heatmap
    st.plotly_chart(heatmap, use_container_width=True)

if __name__ == '__main__':
    main()

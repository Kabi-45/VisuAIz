import streamlit as st
from Classification import ClassificationPage
from Regression import RegressionPage

class Application:
    def __init__(self):
        """Main entry point for the Streamlit app."""
        self.classification_page = ClassificationPage()
        self.regression_page = RegressionPage()


    def show_home(self):
        """Display the home page."""
        st.title("Welcome to VisuAIz")
        st.write("""
        This app allows you to explore and visualize machine learning models for **Classification** and **Regression**.

        **Features:**
        - Choose preloaded datasets or upload a CSV
        - Select from various machine learning models
        - Tune hyperparameters dynamically
        - Visualize model performance

        Use the sidebar to navigate!
        """)

    def show_classification_page(self):
        """Display the classification page."""
        self.classification_page.run()

    def show_regression_page(self):
        """Display the regression page."""
        self.regression_page.run()

    def run(self):
        """Main navigation logic for the app."""
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Home", "Classification", "Regression"])

        if page == "Home":
            self.show_home()
        elif page == "Classification":
            self.show_classification_page()
        elif page == "Regression":
            self.show_regression_page()

# Run the app
if __name__ == "__main__":
    app = Application()
    app.run()

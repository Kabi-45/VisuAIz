import streamlit as st
from dataset import DatasetLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from visualize import Visualization

class RegressionPage:
    def __init__(self):
        """Initialize the Regression Model class."""
        self.dataset_loader = DatasetLoader()
        self.model = None
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def show_sidebar(self):
        """Handle dataset and model selection from sidebar."""
        st.sidebar.title("Regression Settings")
        dataset_name = st.sidebar.selectbox("Select Dataset", ["Diabetes", "Custom CSV"])
        self.X, self.y = self.dataset_loader.load_dataset(dataset_name)

        regressor_name = st.sidebar.selectbox("Select Regressor",
                                              ["Linear Regression", "Ridge Regression", "Lasso Regression"])
        self.set_model(regressor_name)

    def set_model(self, regressor_name):
        """Set the regression model based on user selection."""
        params = {}
        if regressor_name in ["Ridge Regression", "Lasso Regression"]:
            params["alpha"] = st.sidebar.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0)

        if regressor_name == "Linear Regression":
            self.model = LinearRegression()
        elif regressor_name == "Ridge Regression":
            self.model = Ridge(alpha=params["alpha"])
        elif regressor_name == "Lasso Regression":
            self.model = Lasso(alpha=params["alpha"])

    def train_and_evaluate(self):
        """Train the selected model and display results."""
        if self.X is not None and self.y is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.model.fit(self.X_train, self.y_train)
            y_pred = self.model.predict(self.X_test)

            mse = mean_squared_error(self.y_test, y_pred)
            st.write(f"### Mean Squared Error: {mse:.2f}")

            if self.X.shape[1] == 1:
                Visualization.plot_regression_line(y_pred)
            Visualization.plot_regression_results(self.y_test, y_pred)
            # Visualization.plot_residuals(self.y_test, y_pred)

    def run(self):
        """Main entry point for the Regression page."""
        st.title("Regression Model Demo")
        self.show_sidebar()
        self.train_and_evaluate()

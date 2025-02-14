import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
# from sklearn.inspection import permutation_importance


class Visualization:
    """Class for visualizing classification and regression metrics."""

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred):
        """Plot and display a confusion matrix for classification models."""
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    @staticmethod
    def plot_decision_boundary(model, X, y):
        """Plot decision boundaries for classification models with 2D feature space."""
        if X.shape[1] != 2:
            st.warning("Decision boundary visualization is only available for 2D feature spaces.")
            return

        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, Z, alpha=0.3)
        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors="k")
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        st.pyplot(fig)





    @staticmethod
    def plot_regression_line(self, y_pred):
        """Plot the regression line if the dataset has only one feature."""
        st.write("### Regression Line")
        fig, ax = plt.subplots()
        ax.scatter(self.X_test, self.y_test, color="blue", label="Actual")
        ax.plot(self.X_test, y_pred, color="red", linewidth=2, label="Prediction")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
        ax.legend()
        st.pyplot(fig)

    @staticmethod
    def plot_regression_results(y_test, y_pred):
        """Plot regression model performance (scatter plot for actual vs predicted)."""
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color="blue", edgecolors="black", alpha=0.7, label="Predicted vs Actual")
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", linewidth=2,
                label="Perfect Fit")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()
        st.pyplot(fig)

    @staticmethod
    def plot_residuals(y_test, y_pred):
        """Plot residuals for regression models."""
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals, color="purple", alpha=0.7)
        ax.axhline(y=0, color="red", linestyle="--")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        st.pyplot(fig)

    @staticmethod
    def plot_feature_importance(model, X_train, y_train):
        """Plot feature importance for tree-based models (like Random Forest)."""
        if not hasattr(model, "feature_importances_"):
            st.warning("Feature importance is only available for tree-based models (e.g., Random Forest).")
            return

        importance = model.feature_importances_
        feature_names = X_train.columns
        sorted_indices = np.argsort(importance)

        fig, ax = plt.subplots()
        ax.barh(range(len(importance)), importance[sorted_indices], align="center")
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels(np.array(feature_names)[sorted_indices])
        ax.set_xlabel("Feature Importance Score")
        ax.set_title("Feature Importance")
        st.pyplot(fig)


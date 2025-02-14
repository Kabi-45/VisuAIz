import streamlit as st
from dataset import DatasetLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from visualize import Visualization

class ClassificationPage:
    def __init__(self):
        """Initialize the Classification Model class."""
        self.dataset_loader = DatasetLoader()
        self.model = None
        self.classifier_name = None
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def show_sidebar(self):
        """Handle dataset and model selection from sidebar."""
        st.sidebar.title("Classification Settings")
        dataset_name = st.sidebar.selectbox("Select Dataset", ["Iris", "Custom CSV"])
        self.X, self.y = self.dataset_loader.load_dataset(dataset_name)

        classifier_name = st.sidebar.selectbox("Select Classifier", ["Logistic Regression", "SVM", "Random Forest"])
        self.set_model(classifier_name)

    def set_model(self, classifier_name):
        """Set the classifier based on user selection."""
        params = {}
        if classifier_name == "Logistic Regression":
            params["C"] = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
            self.model = LogisticRegression(C=params["C"])
        elif classifier_name == "SVM":
            params["C"] = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
            params["Kernel"] = st.sidebar.radio("Kernel", ["linear", "rbf", "poly"])
            self.model = SVC(C=params["C"], kernel=params["Kernel"])
        elif classifier_name == "Random Forest":
            params["n_estimators"] = st.sidebar.slider("Number of Trees", 10, 200, 100)
            params["max_depth"] = st.sidebar.slider("Max Depth", 2, 20, 5)
            self.model = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"])
        self.classifier_name = classifier_name


    def train_and_evaluate(self):
        """Train the selected model and display results."""
        if self.X is not None and self.y is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.model.fit(self.X_train, self.y_train)
            y_pred = self.model.predict(self.X_test)

            metric = st.selectbox("Select Metrics", ["Model Accuracy","Confusion Matrix"])
            if metric == "Model Accuracy":
                accuracy = accuracy_score(self.y_test, y_pred)
                st.write(f"### Model Accuracy: {accuracy:.2f}")
            elif metric == "Confusion Matrix":
                st.write("### Confusion Matrix")
                Visualization.plot_confusion_matrix(self.y_test, y_pred)
                if self.classifier_name == "Random Forest":
                    Visualization.plot_feature_importance(self.model, self.X_train, self.y_train)


            if self.X.shape[1] == 2:
                st.write("### Decision Boundary")
                Visualization.plot_decision_boundary(self.model, self.X_test, self.y_test)

            # if hasattr(self.model, "predict_proba"):
            #     y_pred_probs = self.model.predict_proba(self.X_test)  # Keep full probability matrix for multiclass
            #     classes = np.unique(self.y_test)  # Extract unique classes
            #     Visualization.plot_roc_curve(self.y_test, y_pred_probs, classes)

            # Feature importance for RandomForest

    def run(self):
        """Main entry point for the Classification page."""
        st.title("Classification Model Demo")
        self.show_sidebar()
        self.train_and_evaluate()

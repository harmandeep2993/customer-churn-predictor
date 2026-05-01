from setuptools import setup, find_packages

setup(
    name="customer_churn_prediction",
    version="1.0.0",
    author="Harmandeep",
    author_email="harmandeep2993@example.com",
    description="End-to-end customer churn prediction project using XGBoost and Streamlit.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib",
        "pyyaml",
        "streamlit",
        "matplotlib",
        "seaborn"
    ],
    python_requires=">=3.12.10",
)
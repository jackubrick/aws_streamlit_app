# AWS Streamlit Application

### Create Virtual Environment
```
conda create --prefix ./venv python=3.9.16
```
```
conda activate C:\repos\aws_streamlit_app\venv
```

### pip freeze all dependencies
```
pip freeze > data/requirements_frozen.txt
```

### Project Structure
```
aws_streamlit_app/
├── ui/                  # All front-end related code
├── backend/             # Backend-related code, services, and APIs
├── vectorstore/         # Vector database or store-related files
├── .env                 # Environment variables
├── .gitignore           # Git ignore file
├── README.md            # Project documentation
├── requirements.txt     # Dependencies for the project
├── run.py               # Entry point for running the app
└── tests/               # Test folder
```


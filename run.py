import subprocess

def run_streamlit_app(script_path):
    subprocess.call(["streamlit", "run", script_path, "--server.headless", "true"])

if __name__ == "__main__":
    # Because root is Gen_AI:
    run_streamlit_app("ui/RAG.py")
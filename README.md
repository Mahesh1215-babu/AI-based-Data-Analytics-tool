📊 AI-Powered Data Analytics Pipeline Tool
🚀 Live Demo
Experience the full pipeline directly on the web!

Website URL: http://localhost:8501

✨ Project Overview
This is a comprehensive, multi-step data analytics application built with Streamlit and powered by Pandas, Matplotlib, Seaborn, Scikit-learn, and the Groq AI API.

The tool guides users through a structured data workflow, from raw ingestion to AI-assisted analysis and forecasting, mimicking a professional data science pipeline.

🛠️ Key Features & Pipeline Steps
The application is structured into eight distinct steps, accessible via the sidebar navigation:

Step	Title	Description	Core Technology
1	Data Ingestion	Upload CSV or Excel files to begin the analysis.	Pandas
2	Cleaning	Automated handling of missing values (ffill/bfill), duplicate removal, and type conversions.	Pandas
3	EDA (Exploratory Data Analysis)	Generates summary statistics (describe()) and visualizes missing data distributions.	Pandas
4	AI Enhancement	Demonstrates data preprocessing/feature engineering, such as Min-Max Scaling on a selected column.	NumPy / Scikit-learn
5	Visualization	A Power BI-style chart builder interface allowing users to select chart types and corresponding data fields (axes/values) to generate custom visualizations.	Matplotlib, Seaborn, Streamlit Layouts
6	Data-Aware Chatbot	An interactive AI assistant that can answer general knowledge questions AND provide analysis and insights on the uploaded data using Groq's fast LLMs.	Groq API (Gemma2-9b-it)
7	Forecasting	A simple time-series forecasting model using Linear Regression to predict future trends based on a selected datetime and value column.	Scikit-learn
8	Export/Dashboard	Final view of the processed data and options to download the clean, enhanced dataset as CSV or Excel.	Pandas

Export to Sheets
⚙️ Setup and Installation
Prerequisites
Python 3.8+

Groq API Key: Required for the Data-Aware Chatbot functionality (Step 6). Get one from the Groq console.

Local Installation
Clone the repository:

Bash

git clone https://github.com/Mahesh1215-babu/ai-data-analytics-tool.git
cd ai-data-analytics-tool
Create and activate a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate   # On Windows
Install dependencies:
The core dependencies are listed below. You should run:

Bash

pip install streamlit pandas numpy matplotlib seaborn scikit-learn groq openpyxl
Set up the API Key:
Create a folder named .streamlit in your project root and inside it, create a file named secrets.toml. Add your Groq API key:

Ini, TOML

# .streamlit/secrets.toml
GROQ_API_KEY="sk_gq_[YOUR_GROQ_API_KEY]"
Run the application:

Bash

streamlit run app.py
The application will open in your web browser, typically at http://localhost:8501.

📝 Dependencies
The primary dependencies used in this project are:

streamlit

pandas

numpy

matplotlib

seaborn

scikit-learn

groq

openpyxl (for Excel support)

flask (If you have a separate backend API)

flask_cors (If you have a separate backend API)

👤 Author
Mahesh Babu

GitHub: Mahesh1215-babu

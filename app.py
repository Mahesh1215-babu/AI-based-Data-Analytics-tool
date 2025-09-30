import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from io import BytesIO
from sklearn.linear_model import LinearRegression
from datetime import date 
import os
from groq import Groq

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="AI Data Analytics Tool", layout="wide")
st.sidebar.title("📌 Pipeline Steps")

# Define all pages in the pipeline
PAGES = [
    "Home",
    "1. Data Ingestion",
    "2. Cleaning",
    "3. EDA",
    "4. AI Enhancement",
    "5. Visualization", # Target page for change
    "6. Data-Aware Chatbot", 
    "7. Forecasting",
    "8. Export/Dashboard",
]

# Initialize session state for all variables upfront for clarity
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "df" not in st.session_state:
    st.session_state["df"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "general_chat_history" not in st.session_state: # History for the dedicated chatbot
    st.session_state["general_chat_history"] = []

# Sidebar navigation
page = st.sidebar.radio(
    "Navigate", 
    PAGES, 
    index=PAGES.index(st.session_state.get("page", "Home")),
    key="pipeline_navigation"
)

st.session_state["page"] = page 

# ---------------------------
# DATA-AWARE GROQ CHAT FUNCTION 
# ---------------------------

def groq_general_chat(query: str, history: list, df: pd.DataFrame = None) -> str:
    """Uses the Groq API for conversational response, now with optional data context."""
    
    if "GROQ_API_KEY" not in st.secrets:
        return "❌ Error: Groq API key not found in Streamlit secrets. Conversational AI is disabled."

    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception as e:
         return f"❌ Groq API Client Error: Failed to initialize. Check your API key. Details: {e}"

    # Generate Data Context
    system_prompt = "You are a helpful, concise AI assistant. Keep your answers brief."
    
    if df is not None and not df.empty:
        # Create metadata string to include in the prompt
        df_head = df.head(3).to_markdown(index=False)
        df_info = df.describe(include='all').T.to_markdown()

        system_prompt += (
            "\n\n*** DATA CONTEXT FOR ANALYSIS ***"
            "\n- The user has uploaded a dataset."
            f"\n- Total Rows: {len(df):,}, Total Columns: {len(df.columns)}"
            "\n- Column Names and Data Types (df.info()):\n" + df.dtypes.to_string() +
            "\n- Summary Statistics (df.describe().T):\n" + df_info +
            "\n- First 3 Rows (df.head()):\n" + df_head +
            "\n*** END DATA CONTEXT ***"
            "\n\n**If the user asks a question about the data, use the context provided above to answer. If the question is general, ignore the context.**"
        )

    # Build Message History
    messages = [{"role": "system", "content": system_prompt}]
    
    for chat in history[-6:]: 
        messages.append({"role": "user", "content": chat["query"]})
        messages.append({"role": "assistant", "content": chat["response"]})

    messages.append({"role": "user", "content": query})

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gemma2-9b-it", 
            temperature=0.7, 
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Groq API Error: Failed to get response. Details: {e}"


# ---------------------------
# HOME PAGE
# ---------------------------
if page == "Home":
    st.title("📊 AI-Powered Data Analytics Tool")
    st.markdown("""
    ### Welcome! 🚀
    This tool follows a **complete AI-driven data analytics pipeline**:

    1. **Data Ingestion** → Upload CSV/Excel data 💾
    2. **Cleaning** → Handle missing values, duplicates, type fixes 🧹
    3. **EDA** → Auto summary & distributions 🔍
    4. **AI Enhancement** → Smart preprocessing (e.g., Feature Engineering, Encoding) 🤖
    5. **Visualization** → Interactive charts & graphs 📈
    6. **Data-Aware Chatbot** → Analyze your data or ask general questions! 💬
    7. **Forecasting** → Predict future trends 🔮
    8. **Export/Dashboard** → Save results or dashboards 📤

    ---
    """)
    if st.button("🚀 Start Pipeline (Go to Data Ingestion)", type="primary"):
        st.session_state["page"] = "1. Data Ingestion"
        st.rerun()

# ---------------------------
# 1. DATA INGESTION
# ---------------------------
elif page == "1. Data Ingestion":
    st.header("📥 Data Ingestion")
    uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            with st.spinner(f"Loading {uploaded_file.name}..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file, encoding='utf8', on_bad_lines='skip', engine='python')
                else:
                    df = pd.read_excel(uploaded_file)
            
            st.session_state["df"] = df
            # Reset chat histories when new data is uploaded
            st.session_state["chat_history"] = [] 
            st.session_state["general_chat_history"] = [] 
            st.success("✅ Data uploaded successfully! (First 5 rows displayed)")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file. Please check file format and encoding. Error: {e}")

# ---------------------------
# 2. CLEANING
# ---------------------------
elif page == "2. Cleaning":
    st.header("🧹 Data Cleaning")

    if st.session_state["df"] is not None:
        df = st.session_state["df"].copy()
        initial_rows = len(df)
        
        with st.expander("ℹ️ Cleaning Steps Applied"):
             st.markdown("""
             - **Missing Values:** Forward fill (`ffill`) followed by backward fill (`bfill`).
             - **Duplicates:** Rows with identical data across all columns are dropped.
             - **Type Conversion:** Object columns that contain only numbers are forcibly converted to numeric.
             """)
        
        # --- Cleaning Logic ---
        df = df.ffill().bfill()
        df = df.drop_duplicates()
        final_rows = len(df)

        converted_cols = []
        for col in df.select_dtypes(include="object").columns:
            test_numeric = pd.to_numeric(df[col], errors='coerce')
            if test_numeric.isnull().sum() < (len(df) * 0.1): 
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    converted_cols.append(col)
                except:
                    pass
        # --- End Cleaning Logic ---

        st.session_state["df"] = df
        
        st.success("✅ Data cleaning complete!")
        st.markdown(f"**Rows Removed (Duplicates):** {initial_rows - final_rows:,}")
        if converted_cols:
            st.info(f"**Columns Converted to Numeric:** {', '.join(converted_cols)}")
        else:
            st.info("No object columns were converted to numeric.")

        st.dataframe(df.head())
    else:
        st.warning("⚠️ Please upload data first on the '1. Data Ingestion' page.")

# ---------------------------
# 3. EDA
# ---------------------------
elif page == "3. EDA":
    st.header("🔍 Exploratory Data Analysis (EDA)")

    if st.session_state["df"] is not None:
        df = st.session_state["df"]

        st.subheader("📊 Data Summary")
        st.dataframe(df.describe(include="all").T)

        st.subheader("📈 Missing Values Check")
        missing_data = df.isnull().sum()
        missing_data_to_plot = missing_data[missing_data > 0] 
        
        if missing_data_to_plot.sum() > 0:
            st.warning("⚠️ Missing values detected! (Imputed in Cleaning step)")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            missing_data_to_plot.plot(kind='bar', ax=ax, title='Missing Values by Column (Before Imputation)')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            with st.expander("Details"):
                 st.dataframe(pd.DataFrame(missing_data).rename(columns={0:'Missing Count'}).sort_values('Missing Count', ascending=False))
        else:
            st.success("✅ No missing values found in the current dataset! (All were handled in the Cleaning step)")
    else:
        st.warning("⚠️ Please upload data first on the '1. Data Ingestion' page.")

# ---------------------------
# 4. AI ENHANCEMENT
# ---------------------------
elif page == "4. AI Enhancement":
    st.header("🤖 AI Enhancement (Demo)")
    st.markdown("This section simulates advanced AI preprocessing like normalization, scaling, or feature engineering.")
    
    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if numeric_cols:
             st.markdown("---")
             st.subheader("🔬 Min-Max Scaling Example")
             
             col_to_scale = st.selectbox("Select a numeric column for Min-Max Scaling (Demo):", numeric_cols)
             
             if st.button("Apply Scaling (Add New Column)"):
                 col_data = df[col_to_scale].dropna()
                 if not col_data.empty and col_data.max() != col_data.min():
                     scaled_data = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                     new_col_name = f"{col_to_scale}_Scaled"
                     df[new_col_name] = scaled_data
                     st.session_state["df"] = df
                     st.success(f"✅ Added new column: '{new_col_name}'. Data scaled between 0 and 1.")
                 else:
                     st.warning("Cannot scale: Column is empty, has only one unique value, or contains NaNs.")
                     
             st.dataframe(df.head())
        else:
            st.info("No numeric columns available for scaling demonstration.")
        
    else:
        st.warning("⚠️ Please upload data first on the '1. Data Ingestion' page.")

# ---------------------------
# 5. VISUALIZATION (UPDATED TO POWER BI BUILDER STYLE)
# ---------------------------
elif page == "5. Visualization":
    st.header("📊 Visualization Builder")
    st.markdown("Use the controls in the sidebar to build your chart, similar to the Power BI Visualizations pane.")

    if st.session_state["df"] is None:
        st.warning("⚠️ Please upload data first on the '1. Data Ingestion' page.")
        
    else:
        df = st.session_state["df"]
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # --- Sidebar Content (Mimicking Power BI Pane) ---
        with st.sidebar:
            st.subheader("🛠️ Build Visual")
            
            # 1. Chart Selection (Mimics visual icons)
            chart_type = st.selectbox(
                "Select Chart Type", 
                ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Area Chart", "Correlation Heatmap"],
                index=0, 
                key="builder_chart_type"
            )
            st.markdown("---")
            
            st.subheader("Data Fields")
            
            # 2. Data Field Wells (Mimics Values/Axis/Legend)
            if chart_type in ["Bar Chart", "Line Chart", "Area Chart"]:
                x_col = st.selectbox("X-Axis (Category/Date)", all_cols, index=all_cols.index(categorical_cols[0]) if categorical_cols else 0, key="x_axis_col")
                y_col = st.selectbox("Y-Axis (Value)", numeric_cols, index=0 if numeric_cols else 0, key="y_axis_col")
                
            elif chart_type == "Scatter Plot":
                x_col = st.selectbox("X-Axis (Numeric)", numeric_cols, index=0 if numeric_cols else 0, key="scatter_x_col")
                y_col = st.selectbox("Y-Axis (Numeric)", [c for c in numeric_cols if c != x_col], index=0 if len(numeric_cols) > 1 else 0, key="scatter_y_col")
            
            elif chart_type == "Histogram":
                hist_col = st.selectbox("Value to Distribute", numeric_cols, index=0 if numeric_cols else 0, key="hist_col")
                
            elif chart_type == "Correlation Heatmap":
                st.info("Uses all numeric columns automatically.")

        # --- Main Area (Displaying the Chart) ---
        
        if not all_cols:
             st.error("Dataset is empty.")
             
        elif chart_type in ["Bar Chart", "Line Chart", "Area Chart"]:
            if x_col and y_col:
                # Basic aggregation (sum) for the visual
                plot_data = df.groupby(x_col)[y_col].sum().reset_index()
                
                st.subheader(f"{chart_type} of {y_col} by {x_col}")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if chart_type == "Bar Chart":
                    sns.barplot(x=x_col, y=y_col, data=plot_data, ax=ax)
                elif chart_type == "Line Chart":
                    sns.lineplot(x=x_col, y=y_col, data=plot_data, ax=ax)
                elif chart_type == "Area Chart":
                    ax.fill_between(plot_data[x_col], plot_data[y_col], alpha=0.5)
                    ax.plot(plot_data[x_col], plot_data[y_col])
                    
                ax.set_title(f'{chart_type}: Sum of {y_col} by {x_col}')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            else:
                 st.warning("Please select valid X and Y axis columns.")
                 
        elif chart_type == "Scatter Plot":
            if x_col and y_col and x_col != y_col:
                st.subheader(f"Scatter Plot of {y_col} vs {x_col}")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
                ax.set_title(f'Scatter Plot: {y_col} vs {x_col}')
                st.pyplot(fig)
            else:
                st.warning("Please select two distinct numeric columns for the scatter plot.")
                
        elif chart_type == "Histogram":
            if hist_col:
                st.subheader(f"Distribution of {hist_col}")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[hist_col].dropna(), kde=True, ax=ax)
                ax.set_title(f'Histogram of {hist_col}')
                st.pyplot(fig)
            else:
                st.warning("Please select a numeric column for the histogram.")
                
        elif chart_type == "Correlation Heatmap":
            if len(numeric_cols) >= 2:
                st.subheader("Correlation Heatmap (All Numeric Columns)")
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar_kws={'shrink': 0.75})
                ax.set_title('Correlation Heatmap')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                st.pyplot(fig)
            else:
                st.warning("Need at least two numeric columns for a correlation heatmap.")


# ---------------------------
# 6. DATA-AWARE CHATBOT 
# ---------------------------
elif page == "6. Data-Aware Chatbot":
    st.header("💬 AI Data Analyst Chatbot")
    
    current_df = st.session_state["df"]
    
    if current_df is not None and not current_df.empty:
        st.success(f"✅ Chatbot is connected to your data! ({len(current_df):,} rows, {len(current_df.columns)} columns).")
        st.markdown("Ask for summaries, column explanations, or general facts.")
        
        with st.expander("ℹ️ Data Context Provided to AI"):
            st.write("The AI receives summary statistics, data types, and the first few rows of your current dataset for context.")
            st.dataframe(current_df.head(2))
    else:
        st.info("⚠️ No data uploaded. The chatbot is currently a **General AI Chatbot** only.")
        st.markdown("Ask the AI any general question (e.g., 'What is machine learning?').")


    # Display chat history 
    for chat in st.session_state.general_chat_history:
        with st.chat_message("user"):
            st.write(chat["query"])
        with st.chat_message("assistant"):
            st.write(chat["response"])
    
    # User input
    user_query = st.chat_input("Ask about your data or a general question...", key="data_chat_input")
    
    if user_query:
        # Pass the DataFrame to the chat function
        with st.spinner("🤖 Thinking..."):
            response = groq_general_chat(
                user_query, 
                st.session_state.general_chat_history,
                df=current_df # Pass the dataframe here
            )
        
        # Save to chat history
        st.session_state.general_chat_history.append({
            "query": user_query,
            "response": response,
        })
        
        st.rerun() 
        
    st.markdown("---")
    # Clear chat button
    if st.button("🗑️ Clear Chat History", key="clear_data_chat"):
        st.session_state.general_chat_history = []
        st.rerun()

# ---------------------------
# 7. FORECASTING
# ---------------------------
elif page == "7. Forecasting":
    st.header("📈 Forecasting")

    if st.session_state["df"] is not None:
        df = st.session_state["df"].copy()
        
        st.info(f"📋 Dataset columns: {', '.join(df.columns.tolist())}")

        # --- Datetime Column Detection ---
        datetime_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day', 'timestamp']):
                datetime_cols.append(col)
            elif df[col].dtype == 'object':
                try:
                    temp_parse = pd.to_datetime(df[col].sample(min(100, len(df)), random_state=42), errors='coerce')
                    if temp_parse.notna().sum() / min(100, len(df)) > 0.8: 
                        if col not in datetime_cols:
                            datetime_cols.append(col)
                except Exception:
                    pass
        
        if datetime_cols:
            date_col = st.selectbox("Select date/time column:", list(set(datetime_cols)))
            
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if not numeric_cols:
                st.error("⚠️ No numeric column found for forecasting.")
            else:
                val_col = st.selectbox("Select value column to forecast:", numeric_cols)
                
                ts = df[[date_col, val_col]].dropna().sort_values(date_col)
                ts = ts.set_index(date_col).groupby(level=0).mean() 
                
                if len(ts) < 2:
                    st.error("⚠️ Not enough unique, valid data points to perform a forecast.")
                else:
                    st.subheader("📊 Historical Data")
                    st.line_chart(ts)
                    st.write(f"Data points: {len(ts):,}")

                    forecast_steps = st.slider("Forecast steps:", 5, 60, 10, key="forecast_steps")
                    
                    ts_reset = ts.reset_index(names=[date_col])
                    ts_reset["t"] = range(len(ts_reset))
                    
                    train_data = ts_reset.dropna(subset=[val_col])
                    
                    if len(train_data) == 0:
                        st.error("⚠️ Value column contains only NaN values. Cannot train model.")
                    else:
                        model = LinearRegression().fit(train_data[["t"]], train_data[val_col])
                        
                        last_date = train_data[date_col].iloc[-1]
                        inferred_freq = pd.infer_freq(train_data[date_col])
                        
                        if inferred_freq is None:
                            st.warning("⚠️ Could not reliably infer date frequency. Defaulting to 'Day'.")
                            inferred_freq = 'D'

                        future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=inferred_freq)[1:]
                        
                        future_t = pd.DataFrame({"t": range(len(train_data), len(train_data) + forecast_steps)})
                        future_t[val_col] = model.predict(future_t[["t"]])
                        
                        future_t['Date'] = future_dates
                        future_t = future_t.set_index('Date')[[val_col]]
                        
                        combined_data = pd.concat([ts, future_t], axis=0)
                        
                        st.subheader(f"🔮 Forecast (next {forecast_steps} steps)")
                        st.dataframe(future_t)
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ts.plot(ax=ax, label='Historical Data', color='blue')
                        future_t.plot(ax=ax, label='Forecast', color='red', linestyle='--')
                        
                        ax.set_title(f'Historical vs. Forecasted {val_col}')
                        ax.legend()
                        st.pyplot(fig)
                        
                        st.info(f"📊 Simple Linear Regression Model R² score: {model.score(train_data[['t']], train_data[val_col]):.4f}")
                        st.caption("A simple Linear Regression is used here for demonstration. Real-world forecasting requires more complex models like ARIMA or Prophet.")

        else:
            st.warning("⚠️ No datetime column detected in your dataset.")
            
            st.markdown("### 🔧 Create Synthetic Date Column for Forecasting")
            st.write("Since your data doesn't have dates, you can create a sequential date column for demonstration:")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date:", date(2024, 1, 1), key="synth_start_date")
            with col2:
                freq = st.selectbox("Frequency:", ["Daily", "Weekly", "Monthly", "Yearly"], key="synth_freq")
            
            if st.button("Generate Date Column"):
                freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Yearly": "YS"}
                date_range = pd.date_range(start=start_date, periods=len(df), freq=freq_map[freq])
                df["Generated_Date"] = date_range
                st.session_state["df"] = df
                st.success(f"✅ Created 'Generated_Date' column with {freq.lower()} frequency!")
                st.info("🔄 Please click the 'Forecasting' page in the sidebar to refresh and use the new date column.")
                st.dataframe(df.head())
            
            st.markdown("""
            ---
            **Alternative: Upload data with a date column**
            """)
            
            with st.expander("🔍 Show current column types"):
                st.write(df.dtypes)
    else:
        st.warning("⚠️ Please upload data first on the '1. Data Ingestion' page.")

# ---------------------------
# 8. EXPORT
# ---------------------------
elif page == "8. Export/Dashboard":
    st.header("📤 Export Results")

    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.markdown(f"The current processed dataset has **{len(df):,}** rows and **{len(df.columns)}** columns.")
        
        # CSV Export
        buffer_csv = BytesIO()
        df.to_csv(buffer_csv, index=False)
        buffer_csv.seek(0)
        st.download_button(
            label="Download Processed Data as CSV",
            data=buffer_csv.getvalue(),
            file_name="processed_data.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Excel Export
        buffer_excel = BytesIO()
        with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
             df.to_excel(writer, sheet_name='Processed_Data', index=False)
        
        st.download_button(
            label="Download Processed Data as Excel",
            data=buffer_excel.getvalue(),
            file_name="processed_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.markdown("---")
        st.subheader("🖼️ Dashboard Simulation")
        st.info("This is a placeholder for a full dashboard. You can add static visualizations here for a final view.")
        
        if len(df.select_dtypes(include=np.number).columns) > 0:
            st.subheader("Sample Trend Line")
            st.line_chart(df.select_dtypes(include=np.number).head(50))
        
        st.success("✅ Ready to download your final processed data!")
    else:
        st.warning("⚠️ Please upload data first on the '1. Data Ingestion' page.")
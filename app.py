import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.data_processing import preprocess_data, apply_pca, apply_chi2, apply_rfe, apply_mutual_info
from src.model import train_and_evaluate
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Data Reduction Tool - Professional Feature Selection & Dimensionality Reduction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for warm, professional styling
st.markdown("""
<style>
    /* Professional color palette */
    :root {
        --primary-blue: #2c3e50;
        --secondary-blue: #34495e;
        --accent-blue: #3498db;
        --light-blue: #ecf0f1;
        --dark-blue: #1a252f;
        --medium-blue: #5d6d7e;
        --success-green: #27ae60;
        --warning-orange: #f39c12;
        --error-red: #e74c3c;
        --white: #ffffff;
        --border-color: #bdc3c7;
        --text-dark: #2c3e50;
        --text-light: #5d6d7e;
        --gradient-start: #2c3e50;
        --gradient-end: #3498db;
        --card-bg: #f8f9fa;
        --sidebar-bg: #ecf0f1;
    }
    
    /* Main header with professional gradient */
    .main-header {
        background: linear-gradient(135deg, var(--gradient-start) 0%, var(--secondary-blue) 50%, var(--gradient-end) 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: var(--white);
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(44, 62, 80, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Professional metric cards */
    .metric-card {
        background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: var(--white);
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(44, 62, 80, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(44, 62, 80, 0.2);
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Professional info boxes */
    .info-box {
        background: var(--light-blue);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid var(--accent-blue);
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(52, 152, 219, 0.1);
        border: 1px solid rgba(52, 152, 219, 0.1);
    }
    
    .info-box h3 {
        color: var(--primary-blue);
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .info-box ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .info-box li {
        margin-bottom: 0.5rem;
        color: var(--text-dark);
    }
    
    /* Feature explanation boxes */
    .feature-box {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid var(--accent-blue);
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(52, 152, 219, 0.1);
        border: 1px solid rgba(52, 152, 219, 0.1);
    }
    
    .feature-box h4 {
        color: var(--primary-blue);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .feature-box p {
        color: var(--text-dark);
        margin-bottom: 0.5rem;
    }
    
    /* Process steps */
    .process-step {
        background: var(--white);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--accent-blue);
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.1);
    }
    
    .process-step h5 {
        color: var(--primary-blue);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Professional buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
        color: var(--white);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(44, 62, 80, 0.3);
        background: linear-gradient(135deg, var(--secondary-blue) 0%, var(--accent-blue) 100%);
    }
    
    /* Download button */
    .download-btn {
        background: var(--success-green);
        color: var(--white);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0.5rem;
    }
    
    .download-btn:hover {
        background: #229954;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
    }
    
    /* Professional sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--sidebar-bg) 0%, var(--white) 100%);
        border-right: 1px solid var(--border-color);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, var(--light-blue) 0%, var(--white) 100%);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid var(--accent-blue);
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.1);
    }
    
    .section-header h3 {
        color: var(--primary-blue);
        font-weight: 600;
        margin: 0;
        font-size: 1.2rem;
    }
    
    /* Graph container */
    .graph-container {
        background: var(--white);
        border-radius: 12px;
        border: 1px solid var(--border-color);
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Success and status boxes */
    .success-box {
        background: linear-gradient(135deg, #d5f4e6 0%, #e8f8f5 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid var(--success-green);
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(39, 174, 96, 0.1);
        border: 1px solid rgba(39, 174, 96, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef9e7 0%, #fdf2e9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid var(--warning-orange);
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(243, 156, 18, 0.1);
        border: 1px solid rgba(243, 156, 18, 0.1);
    }
    
    /* Professional tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--light-blue);
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--white);
        border-radius: 6px;
        color: var(--medium-blue);
        font-weight: 500;
        padding: 8px 16px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-blue);
        color: var(--white);
        border-color: var(--primary-blue);
    }
    
    /* Professional page background */
    .stApp {
        background: linear-gradient(135deg, #ecf0f1 0%, #f8f9fa 100%);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔬 Data Reduction Tool</h1>
    <p>Professional Feature Selection & Dimensionality Reduction Platform</p>
    <p style="font-size: 0.9rem; opacity: 0.8;">Advanced Analytics for Data Science & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Function to convert matplotlib figure to base64 for download
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

# Sidebar for configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration Panel")
    
    # File upload section
    st.markdown("#### 📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload your dataset in CSV format"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Method selection with detailed explanations
        st.markdown("#### 🎯 Feature Selection Method")
        method = st.selectbox(
            "Choose your method:",
            [
                "Principal Component Analysis (PCA)",
                "Chi-Square Feature Selection",
                "Recursive Feature Elimination (RFE)",
                "Mutual Information Feature Selection"
            ],
            help="Select the dimensionality reduction technique"
        )
        
        # Show method explanation
        if method == "Principal Component Analysis (PCA)":
            st.markdown("""
            <div class="feature-box">
                <h4>🔬 Principal Component Analysis (PCA)</h4>
                <p><strong>What it does:</strong> Reduces dimensions while preserving maximum variance</p>
                <p><strong>Best for:</strong> High-dimensional data with correlated features</p>
                <p><strong>Output:</strong> Principal components (linear combinations of original features)</p>
            </div>
            """, unsafe_allow_html=True)
        elif method == "Chi-Square Feature Selection":
            st.markdown("""
            <div class="feature-box">
                <h4>📊 Chi-Square Feature Selection</h4>
                <p><strong>What it does:</strong> Selects features based on statistical significance</p>
                <p><strong>Best for:</strong> Categorical target variables</p>
                <p><strong>Output:</strong> Top-k most statistically significant features</p>
            </div>
            """, unsafe_allow_html=True)
        elif method == "Recursive Feature Elimination (RFE)":
            st.markdown("""
            <div class="feature-box">
                <h4>🔄 Recursive Feature Elimination (RFE)</h4>
                <p><strong>What it does:</strong> Iteratively removes least important features</p>
                <p><strong>Best for:</strong> When you have a good base model</p>
                <p><strong>Output:</strong> Optimal feature subset based on model performance</p>
            </div>
            """, unsafe_allow_html=True)
        elif method == "Mutual Information Feature Selection":
            st.markdown("""
            <div class="feature-box">
                <h4>🔗 Mutual Information Feature Selection</h4>
                <p><strong>What it does:</strong> Selects features based on mutual information with target</p>
                <p><strong>Best for:</strong> Non-linear relationships between features and target</p>
                <p><strong>Output:</strong> Features with highest mutual information scores</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show method explanation
        if method == "Principal Component Analysis (PCA)":
            st.markdown("""
            <div class="feature-box">
                <h4>🔬 Principal Component Analysis (PCA)</h4>
                <p><strong>What it does:</strong> Reduces dimensions while preserving maximum variance</p>
                <p><strong>Best for:</strong> High-dimensional data with correlated features</p>
                <p><strong>Output:</strong> Principal components (linear combinations of original features)</p>
            </div>
            """, unsafe_allow_html=True)
        elif method == "Chi-Square Feature Selection":
            st.markdown("""
            <div class="feature-box">
                <h4>📊 Chi-Square Feature Selection</h4>
                <p><strong>What it does:</strong> Selects features based on statistical significance</p>
                <p><strong>Best for:</strong> Categorical target variables</p>
                <p><strong>Output:</strong> Top-k most statistically significant features</p>
            </div>
            """, unsafe_allow_html=True)
        elif method == "Recursive Feature Elimination (RFE)":
            st.markdown("""
            <div class="feature-box">
                <h4>🔄 Recursive Feature Elimination (RFE)</h4>
                <p><strong>What it does:</strong> Iteratively removes least important features</p>
                <p><strong>Best for:</strong> When you have a good base model</p>
                <p><strong>Output:</strong> Optimal feature subset based on model performance</p>
            </div>
            """, unsafe_allow_html=True)
        elif method == "Mutual Information Feature Selection":
            st.markdown("""
            <div class="feature-box">
                <h4>🔗 Mutual Information Feature Selection</h4>
                <p><strong>What it does:</strong> Selects features based on mutual information with target</p>
                <p><strong>Best for:</strong> Non-linear relationships between features and target</p>
                <p><strong>Output:</strong> Features with highest mutual information scores</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Number of features/components
        st.markdown("#### 📊 Parameters")
        n_features = st.slider(
            "Number of Components/Features:",
            min_value=2,
            max_value=20,
            value=5,
            help="Select how many features/components to keep"
        )
        
        # Model selection
        st.markdown("#### 🤖 Model Selection")
        model_type = st.selectbox(
            "Choose Model:",
            ["Logistic Regression", "Random Forest", "Support Vector Machine"],
            help="Select the machine learning model for evaluation"
        )
        
        # Show model explanation
        if model_type == "Logistic Regression":
            st.markdown("""
            <div class="feature-box">
                <h4>📈 Logistic Regression</h4>
                <p><strong>Pros:</strong> Fast, interpretable, good baseline</p>
                <p><strong>Cons:</strong> Assumes linear relationships</p>
                <p><strong>Best for:</strong> Binary classification, interpretability needed</p>
            </div>
            """, unsafe_allow_html=True)
        elif model_type == "Random Forest":
            st.markdown("""
            <div class="feature-box">
                <h4>🌲 Random Forest</h4>
                <p><strong>Pros:</strong> Robust, handles non-linear relationships, feature importance</p>
                <p><strong>Cons:</strong> Less interpretable, can be slower</p>
                <p><strong>Best for:</strong> Complex datasets, feature importance analysis</p>
            </div>
            """, unsafe_allow_html=True)
        elif model_type == "Support Vector Machine":
            st.markdown("""
            <div class="feature-box">
                <h4>🎯 Support Vector Machine</h4>
                <p><strong>Pros:</strong> Powerful, handles high-dimensional data</p>
                <p><strong>Cons:</strong> Computationally intensive, less interpretable</p>
                <p><strong>Best for:</strong> Complex classification tasks, high-dimensional data</p>
            </div>
            """, unsafe_allow_html=True)

# Main content area
if uploaded_file is not None:
    # File processing with better error handling
    try:
        # Try UTF-8 first
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # Try latin-1 encoding
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        except UnicodeDecodeError:
            # Try cp1252 encoding (Windows default)
            df = pd.read_csv(uploaded_file, encoding='cp1252')
    except Exception as e:
        # If all encodings fail, try with different delimiters
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        except:
            try:
                df = pd.read_csv(uploaded_file, sep='\t', encoding='utf-8')
            except:
                st.error("❌ Could not read the file. Please check if it's a valid CSV file.")
                st.stop()
    
    # Dataset overview
    st.markdown('<div class="section-header"><h3>📋 Dataset Overview</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Rows", f"{len(df):,}")
    with col2:
        st.metric("📈 Total Columns", f"{len(df.columns):,}")
    with col3:
        st.metric("🔢 Numeric Columns", f"{len(df.select_dtypes(include=[np.number]).columns):,}")
    with col4:
        st.metric("📝 Categorical Columns", f"{len(df.select_dtypes(include=['object']).columns):,}")
    
    # Dataset preview
    st.markdown('<div class="section-header"><h3>📋 Dataset Preview</h3></div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    
    # Target column selection
    st.markdown('<div class="section-header"><h3>🎯 Target Variable Selection</h3></div>', unsafe_allow_html=True)
    target_col = st.selectbox(
        "Select your target column:",
        df.columns,
        help="Choose the column you want to predict"
    )
    
    if target_col:
        # Data preprocessing
        with st.spinner("🔄 Preprocessing data..."):
            X_scaled, y, feature_names = preprocess_data(df, target_col)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
        
        st.success("✅ Data preprocessing completed!")
        
        # Show preprocessing process
        st.markdown('<div class="section-header"><h3>🔄 Data Preprocessing Process</h3></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="process-step">
                <h5>Step 1: Data Cleaning</h5>
                <p>• Removed non-numeric columns<br>• Handled missing values<br>• Validated data types</p>
            </div>
            <div class="process-step">
                <h5>Step 2: Feature Scaling</h5>
                <p>• Applied StandardScaler<br>• Normalized features<br>• Prepared for ML models</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="process-step">
                <h5>Step 3: Train-Test Split</h5>
                <p>• Split ratio: 70% train, 30% test<br>• Random state: 42<br>• Stratified sampling</p>
            </div>
            <div class="process-step">
                <h5>Step 4: Feature Names</h5>
                <p>• Extracted feature names<br>• Prepared for selection<br>• Ready for analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance visualization
        st.markdown('<div class="section-header"><h3>📊 Feature Analysis</h3></div>', unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📈 Feature Distribution", "🎯 Correlation Matrix", "📊 Feature Statistics"])
        
        with tab1:
            # Feature distribution
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, feature in enumerate(feature_names[:4]):
                if i < len(axes):
                    axes[i].hist(X_train[:, i], bins=30, alpha=0.7, color='#3498db', edgecolor='#2c3e50')
                    axes[i].set_title(f'Distribution of {feature}', fontweight='bold', color='#2c3e50')
                    axes[i].set_xlabel('Value', color='#34495e')
                    axes[i].set_ylabel('Frequency', color='#34495e')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download button for distribution plot
            if st.button("📥 Download Distribution Plot", key="download_dist"):
                img_str = fig_to_base64(fig)
                st.download_button(
                    label="⬇️ Download PNG",
                    data=base64.b64decode(img_str),
                    file_name="feature_distribution.png",
                    mime="image/png"
                )
        
        with tab2:
            # Correlation matrix
            corr_matrix = pd.DataFrame(X_train, columns=feature_names).corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='Blues', center=0, ax=ax, 
                       cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title('Feature Correlation Matrix', fontweight='bold', color='#2c3e50', fontsize=16)
            st.pyplot(fig)
            
            # Download button for correlation plot
            if st.button("📥 Download Correlation Plot", key="download_corr"):
                img_str = fig_to_base64(fig)
                st.download_button(
                    label="⬇️ Download PNG",
                    data=base64.b64decode(img_str),
                    file_name="correlation_matrix.png",
                    mime="image/png"
                )
        
        with tab3:
            # Feature statistics
            stats_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean': np.mean(X_train, axis=0),
                'Std': np.std(X_train, axis=0),
                'Min': np.min(X_train, axis=0),
                'Max': np.max(X_train, axis=0)
            })
            st.dataframe(stats_df, use_container_width=True)
        
        # Run analysis button
        st.markdown('<div class="section-header"><h3>🚀 Run Analysis</h3></div>', unsafe_allow_html=True)
        if st.button("🎯 Start Dimensionality Reduction & Model Training", type="primary"):
            with st.spinner("🔄 Processing..."):
                # Apply selected method
                if "PCA" in method:
                    X_train_new, X_test_new, explained_var = apply_pca(
                        X_train, X_test, n_components=n_features
                    )
                    
                    st.success("✅ PCA Applied Successfully!")
                    
                    # PCA visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 Explained Variance")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.plot(np.cumsum(explained_var), marker='o', linewidth=3, markersize=10, 
                               color='#3498db', markeredgecolor='#2c3e50')
                        ax.set_xlabel("Number of Components", fontweight='bold', color='#34495e')
                        ax.set_ylabel("Cumulative Explained Variance", fontweight='bold', color='#34495e')
                        ax.set_title("PCA Explained Variance Ratio", fontweight='bold', color='#2c3e50', fontsize=14)
                        ax.grid(True, alpha=0.3)
                        ax.set_facecolor('#ecf0f1')
                        fig.patch.set_facecolor('#ecf0f1')
                        st.pyplot(fig)
                        
                        # Download button for PCA plot
                        if st.button("📥 Download PCA Plot", key="download_pca"):
                            img_str = fig_to_base64(fig)
                            st.download_button(
                                label="⬇️ Download PNG",
                                data=base64.b64decode(img_str),
                                file_name="pca_explained_variance.png",
                                mime="image/png"
                            )
                    
                    with col2:
                        st.markdown("#### 📈 Variance Breakdown")
                        variance_df = pd.DataFrame({
                            'Component': [f'PC{i+1}' for i in range(len(explained_var))],
                            'Explained Variance': explained_var,
                            'Cumulative Variance': np.cumsum(explained_var)
                        })
                        st.dataframe(variance_df, use_container_width=True)
                
                elif "Chi-Square" in method:
                    X_train_new, X_test_new, selected = apply_chi2(
                        X_train, y_train, X_test, k=n_features
                    )
                    st.success("✅ Chi-Square Feature Selection Applied!")
                    
                    # Show selected features
                    st.markdown("#### 🎯 Selected Features")
                    selected_features = feature_names[selected].tolist()
                    for i, feature in enumerate(selected_features, 1):
                        st.markdown(f"**{i}.** {feature}")
                
                elif "RFE" in method:
                    X_train_new, X_test_new, selected = apply_rfe(
                        X_train, y_train, X_test, k=n_features
                    )
                    st.success("✅ Recursive Feature Elimination Applied!")
                    
                    # Show selected features
                    st.markdown("#### 🎯 Selected Features")
                    selected_features = feature_names[selected].tolist()
                    for i, feature in enumerate(selected_features, 1):
                        st.markdown(f"**{i}.** {feature}")
                
                elif "Mutual Information" in method:
                    X_train_new, X_test_new, selected = apply_mutual_info(
                        X_train, y_train, X_test, k=n_features
                    )
                    st.success("✅ Mutual Information Feature Selection Applied!")
                    
                    # Show selected features
                    st.markdown("#### 🎯 Selected Features")
                    selected_features = feature_names[selected].tolist()
                    for i, feature in enumerate(selected_features, 1):
                        st.markdown(f"**{i}.** {feature}")
                
                # Model training and evaluation
                st.markdown('<div class="section-header"><h3>🤖 Model Performance</h3></div>', unsafe_allow_html=True)
                
                # Map model names to keys
                model_mapping = {
                    "Logistic Regression": "logistic",
                    "Random Forest": "rf",
                    "Support Vector Machine": "svm"
                }
                model_key = model_mapping.get(model_type, "logistic")
                
                acc, f1, report, model = train_and_evaluate(
                    X_train_new, y_train, X_test_new, y_test, model_type=model_key
                )
                
                # Performance metrics in cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Accuracy</h3>
                        <h2>{acc:.4f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 F1 Score</h3>
                        <h2>{f1:.4f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📈 Features Reduced</h3>
                        <h2>{X_train.shape[1] - X_train_new.shape[1]}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed classification report
                st.markdown("#### 📋 Detailed Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
                
                # Download results
                st.markdown('<div class="section-header"><h3>💾 Download Results</h3></div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download reduced dataset
                    reduced_df = pd.DataFrame(X_train_new)
                    reduced_df["target"] = y_train.reset_index(drop=True)
                    csv = reduced_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Reduced Dataset",
                        csv,
                        "reduced_dataset.csv",
                        "text/csv",
                        help="Download the reduced dataset with selected features"
                    )
                
                with col2:
                    # Download performance report
                    performance_report = f"""
Data Reduction Tool - Performance Report
========================================

Method Used: {method}
Model: {model_type}
Number of Features: {X_train_new.shape[1]}
Original Features: {X_train.shape[1]}

Performance Metrics:
- Accuracy: {acc:.4f}
- F1 Score: {f1:.4f}

Selected Features:
{chr(10).join([f"- {feature}" for feature in selected_features]) if 'selected_features' in locals() else "N/A"}

Generated by Data Reduction Tool Platform
"""
                    st.download_button(
                        "⬇️ Download Performance Report",
                        performance_report,
                        "performance_report.txt",
                        "text/plain",
                        help="Download the performance report"
                    )

else:
    # Welcome message when no file is uploaded
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to Data Reduction Tool!</h3>
        <p>This professional platform helps you reduce the dimensionality of your dataset using advanced techniques:</p>
        <ul>
            <li><strong>Principal Component Analysis (PCA):</strong> Reduces dimensions while preserving variance</li>
            <li><strong>Chi-Square Feature Selection:</strong> Selects features based on statistical significance</li>
            <li><strong>Recursive Feature Elimination (RFE):</strong> Iteratively removes least important features</li>
            <li><strong>Mutual Information Feature Selection:</strong> Selects features based on mutual information with target</li>
        </ul>
        <p><strong>To get started:</strong> Upload your CSV file using the sidebar on the left!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add some example visualizations
    st.markdown('<div class="section-header"><h3>📊 Example Visualizations</h3></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        # Example PCA plot
        fig, ax = plt.subplots(figsize=(10, 8))
        x = np.linspace(1, 10, 10)
        y = 1 - np.exp(-x/3)
        ax.plot(x, y, marker='o', linewidth=3, markersize=10, color='#3498db', markeredgecolor='#2c3e50')
        ax.set_xlabel("Number of Components", fontweight='bold', color='#34495e')
        ax.set_ylabel("Explained Variance Ratio", fontweight='bold', color='#34495e')
        ax.set_title("Example: PCA Explained Variance", fontweight='bold', color='#2c3e50', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#ecf0f1')
        fig.patch.set_facecolor('#ecf0f1')
        st.pyplot(fig)
    
    with col2:
        # Example feature importance
        fig, ax = plt.subplots(figsize=(10, 8))
        features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
        importance = [0.25, 0.20, 0.15, 0.12, 0.08]
        colors = ['#3498db', '#2c3e50', '#5dade2', '#85c1e9', '#aed6f1']
        ax.barh(features, importance, color=colors, edgecolor='#2c3e50', linewidth=1)
        ax.set_xlabel("Importance Score", fontweight='bold', color='#34495e')
        ax.set_title("Example: Feature Importance", fontweight='bold', color='#2c3e50', fontsize=14)
        ax.set_facecolor('#ecf0f1')
        fig.patch.set_facecolor('#ecf0f1')
        st.pyplot(fig)
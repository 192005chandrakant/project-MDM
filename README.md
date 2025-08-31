# 🔬 Data Reduction Tool

A professional web application for dimensionality reduction and feature selection using cutting-edge machine learning techniques.

## ✨ Features

### 🎯 **Four Advanced Feature Selection Methods**
1. **Principal Component Analysis (PCA)** - Reduces dimensions while preserving maximum variance
2. **Chi-Square Feature Selection** - Selects features based on statistical significance with target
3. **Recursive Feature Elimination (RFE)** - Iteratively removes least important features
4. **Mutual Information Feature Selection** - Selects features based on mutual information with target variable

### 🤖 **Multiple Machine Learning Models**
- **Logistic Regression** - Fast and interpretable for classification tasks
- **Random Forest** - Robust ensemble method with feature importance
- **Support Vector Machine (SVM)** - Powerful kernel-based classification

### 📊 **Rich Visualizations**
- **Feature Distribution Plots** - Histograms showing data distribution
- **Correlation Matrix Heatmap** - Visual representation of feature relationships
- **PCA Explained Variance Plot** - Cumulative variance explained by components
- **Feature Statistics Table** - Comprehensive statistical summary

### 🎨 **Professional UI/UX**
- **Responsive Design** - Works on desktop and mobile devices
- **Interactive Sidebar** - Easy configuration and parameter tuning
- **Real-time Metrics** - Live performance indicators
- **Professional Color Scheme** - Modern blue gradient design
- **Progress Indicators** - Visual feedback during processing

### 💾 **Export Capabilities**
- **Reduced Dataset Download** - CSV format with selected features
- **Performance Report** - Detailed analysis results in text format
- **Graph Downloads** - High-quality PNG exports of all visualizations

## 🚀 Quick Start

### Local Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd streamlit-data-app
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 🌐 **Deploy on Hugging Face Spaces**

### **Step 1: Create Hugging Face Account**
1. Go to [Hugging Face](https://huggingface.co/)
2. Sign up for a free account
3. Verify your email address

### **Step 2: Create a New Space**
1. Click on your profile picture → "New Space"
2. Choose **"Streamlit"** as the SDK
3. Set Space name: `data-reduction-tool`
4. Set visibility: **Public** or **Private**
5. Click "Create Space"

### **Step 3: Upload Your Files**
1. **Upload these files to your Space:**
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `README.md` (documentation)
   - `src/` folder (source code)

2. **File structure should look like:**
```
data-reduction-tool/
├── app.py
├── requirements.txt
├── README.md
└── src/
    ├── data_processing.py
    └── model.py
```

### **Step 4: Configure Space Settings**
1. Go to your Space settings
2. Set **Python version**: 3.9 or higher
3. Set **Hardware**: CPU (free tier) or GPU (if needed)
4. Save settings

### **Step 5: Deploy**
1. Hugging Face will automatically build and deploy your app
2. Wait for the build to complete (usually 2-5 minutes)
3. Your app will be available at: `https://huggingface.co/spaces/YOUR_USERNAME/data-reduction-tool`

## 📖 How to Use

### 1. **Upload Your Data**
- Click "Choose a CSV file" in the sidebar
- Supported formats: CSV files with various encodings (UTF-8, Latin-1, CP1252)
- Automatic delimiter detection (comma, semicolon, tab)

### 2. **Configure Parameters**
- **Select Target Column**: Choose the variable you want to predict
- **Choose Method**: Pick from four feature selection techniques
- **Set Number of Features**: Specify how many features/components to keep
- **Select Model**: Choose the machine learning algorithm

### 3. **Run Analysis**
- Click "Start Dimensionality Reduction & Model Training"
- View real-time progress and results
- Explore interactive visualizations

### 4. **Download Results**
- Download the reduced dataset as CSV
- Get detailed performance reports
- Export high-quality graphs

## 🔧 Technical Details

### **Feature Selection Methods**

#### **Principal Component Analysis (PCA)**
- **Purpose**: Dimensionality reduction while preserving variance
- **Best for**: High-dimensional data with correlated features
- **Output**: Principal components (linear combinations of original features)

#### **Chi-Square Feature Selection**
- **Purpose**: Select features based on statistical significance
- **Best for**: Categorical target variables
- **Output**: Top-k most statistically significant features

#### **Recursive Feature Elimination (RFE)**
- **Purpose**: Iteratively remove least important features
- **Best for**: When you have a good base model
- **Output**: Optimal feature subset based on model performance

#### **Mutual Information Feature Selection**
- **Purpose**: Select features based on mutual information with target
- **Best for**: Non-linear relationships between features and target
- **Output**: Features with highest mutual information scores

### **Machine Learning Models**

#### **Logistic Regression**
- **Pros**: Fast, interpretable, good baseline
- **Cons**: Assumes linear relationships
- **Best for**: Binary classification, interpretability needed

#### **Random Forest**
- **Pros**: Robust, handles non-linear relationships, feature importance
- **Cons**: Less interpretable, can be slower
- **Best for**: Complex datasets, feature importance analysis

#### **Support Vector Machine (SVM)**
- **Pros**: Powerful, handles high-dimensional data
- **Cons**: Computationally intensive, less interpretable
- **Best for**: Complex classification tasks, high-dimensional data

## 📊 Performance Metrics

The application provides comprehensive performance evaluation:

- **Accuracy**: Overall prediction accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Classification Report**: Detailed per-class metrics
- **Feature Reduction**: Number of features removed

## 🛠️ File Structure

```
streamlit-data-app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── src/
    ├── data_processing.py # Feature selection implementations
    └── model.py          # Machine learning models
```

## 🔍 Troubleshooting

### **Common Issues**

1. **File Upload Errors**
   - Ensure your CSV file is properly formatted
   - Check for consistent delimiters
   - Verify encoding (UTF-8 recommended)

2. **Memory Issues**
   - Reduce dataset size for very large files
   - Use fewer features/components
   - Close other applications to free memory

3. **Model Training Errors**
   - Ensure target column has sufficient samples per class
   - Check for missing values in numeric columns
   - Verify target column is categorical for classification

### **Performance Tips**

- **For Large Datasets**: Use PCA or Chi-Square for faster processing
- **For High Accuracy**: Try Random Forest or SVM
- **For Interpretability**: Use Logistic Regression with RFE
- **For Non-linear Relationships**: Use Mutual Information or SVM

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Report bugs and issues
2. Suggest new features
3. Submit pull requests
4. Improve documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [scikit-learn](https://scikit-learn.org/)
- Visualizations with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
- Deployed on [Hugging Face Spaces](https://huggingface.co/spaces)

---

**Happy Data Science! 🎉**
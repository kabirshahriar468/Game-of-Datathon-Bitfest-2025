# BitFest Datathon 2025 - Job Matching Prediction

## 🏆 Competition Overview

This repository contains the solution for the BitFest Datathon 2025 competition on Kaggle. The challenge involves predicting job matching scores based on candidate profiles and job requirements.

**Competition Link:** https://www.kaggle.com/competitions/bitfest-datathon-2025

## 🎯 Problem Statement

The goal is to predict how well a candidate matches a job posting based on various features including:
- Educational background
- Professional experience
- Skills and competencies
- Career objectives
- Certifications
- Language proficiencies

## 📊 Dataset Description

### Features
The dataset contains extensive information about candidates and job postings:

**Candidate Information:**
- `address`: Candidate's location
- `career_objective`: Professional goals and aspirations
- `skills`: Technical and soft skills
- `educational_institution_name`: Name of educational institutions
- `degree_names`: Academic degrees obtained
- `passing_years`: Graduation years
- `educational_results`: Academic performance
- `major_field_of_studies`: Field of study
- `languages`: Language skills
- `proficiency_levels`: Language proficiency levels

**Professional Experience:**
- `professional_company_names`: Previous employers
- `company_urls`: Company websites
- `start_dates` & `end_dates`: Employment duration
- `positions`: Job titles held
- `responsibilities`: Job responsibilities
- `related_skils_in_job`: Skills used in previous roles

**Additional Information:**
- `extra_curricular_activity_types`: Extracurricular activities
- `certification_providers`: Training and certification providers
- `certification_skills`: Certified skills

**Job Requirements:**
- `job_position_name`: Target job title
- `educational_requirements`: Required education level
- `experience_requirement`: Required years of experience
- `age_requirement`: Age requirements
- `skills_required`: Required skills for the position

**Target Variable:**
- `matched_score`: Continuous score representing job match quality (0-1)

### Dataset Statistics
- **Training Set:** 174,479 samples
- **Test Set:** 1,910 samples
- **Features:** 35+ columns with mixed data types (text, numeric, categorical)

## 🔧 Solution Approach

### Team Sirius Solution Overview

Our approach consists of several key components:

#### 1. Exploratory Data Analysis
- **Missing Values Analysis:** Identified columns with high missing value ratios
- **Feature Importance:** Analyzed relevance of different features
- **Data Type Assessment:** Handled mixed data types appropriately
- **Text Data Processing:** Analyzed diverse textual information

#### 2. Data Preprocessing
- **Feature Selection:** Removed columns with >80% missing values
- **Data Extraction:** Extracted meaningful information from text fields
- **Numeric Data Handling:** Converted text-based numeric data to proper format
- **Text Embedding:** Applied various embedding techniques for textual features
- **Hybrid Embedding:** Combined multiple embedding approaches

#### 3. Model Architecture
- **Primary Models:** XGBoost, LightGBM, Neural Networks
- **Ensemble Approach:** Multiple model combinations
- **Cross-Validation:** K-fold validation strategy
- **Hyperparameter Tuning:** Systematic optimization of model parameters

#### 4. Key Techniques
- **TF-IDF Vectorization:** For text feature extraction
- **Feature Engineering:** Created new features from existing data
- **Grouped Embedding:** Specialized embedding for categorical groups
- **Multi-Model Ensemble:** Combined predictions from multiple algorithms

## 📁 File Structure

```
├── README.md                           # This file
├── .gitignore                         # Git ignore rules
├── final-submitted-version.ipynb      # Final competition submission
├── bitmask-rft1.ipynb                # Random Forest approach
├── bitmask-rft2.ipynb                # Enhanced Random Forest
├── bitmask-sk1.ipynb                 # Scikit-learn models
├── bitmask-sk2.ipynb                 # Advanced ML models
├── bitmask-sk3.ipynb                 # Ensemble methods
├── 1.ipynb                           # Initial exploration
├── train_cleaned.csv                 # Preprocessed training data
├── test_cleaned.csv                  # Preprocessed test data
├── sample_submission.csv             # Sample submission format
├── submission_final.csv              # Final submission file
├── submission_final_nn.csv           # Neural network submission
├── submission_final_xnn.csv          # Extended neural network submission
└── submission.csv                    # Alternative submission
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost lightgbm
pip install matplotlib seaborn plotly
pip install nltk transformers
```

### Running the Solution

1. **Data Preprocessing:**
   ```python
   # Load and clean the data
   python data_preprocessing.py
   ```

2. **Model Training:**
   ```python
   # Train individual models
   python train_models.py
   ```

3. **Generate Predictions:**
   ```python
   # Create final submission
   python generate_submission.py
   ```

### Key Notebooks
- `final-submitted-version.ipynb`: Complete end-to-end solution
- `bitmask-rft1.ipynb`: Random Forest baseline
- `bitmask-sk3.ipynb`: Advanced ensemble methods

## 📈 Model Performance

### Validation Results
- **Cross-Validation Score:** Optimized using K-fold validation
- **Evaluation Metric:** Root Mean Squared Error (RMSE)
- **Final Model:** Ensemble of XGBoost, LightGBM, and Neural Networks

### Key Insights
1. **Text Features:** Career objectives and skills are highly predictive
2. **Experience Matching:** Alignment between candidate experience and job requirements
3. **Educational Fit:** Relevance of educational background to job position
4. **Skill Matching:** Overlap between candidate skills and required skills

## 🔍 Feature Engineering Highlights

### Text Processing
- **Skill Extraction:** Automated extraction of technical skills
- **Experience Quantification:** Converting text descriptions to numeric features
- **Semantic Similarity:** Measuring similarity between job requirements and candidate profiles

### Categorical Encoding
- **One-Hot Encoding:** For categorical variables with few categories
- **Target Encoding:** For high-cardinality categorical features
- **Embedding Layers:** For deep learning models

## 📊 Visualization and Analysis

The notebooks include comprehensive visualizations:
- Feature importance plots
- Correlation heatmaps
- Distribution analysis
- Model performance comparisons

## 🤝 Team Information

**Team Name:** Sirius

**Approach:** Multi-model ensemble with advanced feature engineering and text processing techniques.

## 📝 Notes

- The solution emphasizes robust feature engineering and ensemble methods
- Special attention given to handling mixed data types and missing values
- Multiple validation strategies employed to ensure model generalization

## 🏅 Results

Final submission achieved competitive performance through:
- Comprehensive data preprocessing
- Advanced feature engineering
- Multi-model ensemble approach
- Careful validation and hyperparameter tuning

---

*This README provides a comprehensive overview of our BitFest Datathon 2025 solution. For detailed implementation, please refer to the individual notebooks.*

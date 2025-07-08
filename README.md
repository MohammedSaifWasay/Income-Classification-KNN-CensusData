# 💰 Income Classification Using K-Nearest Neighbors (KNN) on U.S. Census Data

## 📌 Abstract
This project aims to classify U.S. citizens into low-income (≤50K) and high-income (>50K) groups using census data. A **K-Nearest Neighbors (KNN)** algorithm was implemented after thorough preprocessing, feature encoding, and hyperparameter tuning. The model achieved an accuracy of **83.7%**, with key predictive features including **education**, **hours worked per week**, and **capital gain**. The results offer data-driven insights that can support policy-making and economic equality efforts.

---

## 📖 Introduction
Understanding income disparity is vital for shaping equitable socio-economic policies. Using a real-world U.S. census dataset, this project:
- Identifies attributes most associated with income level.
- Builds a supervised learning model using **KNN**.
- Evaluates the model’s performance and explores implications of the results.

---

## 🧪 Methods

### 🧹 Data Preprocessing
- Dataset: ~48,842 records and 15 attributes.
- Replaced missing values (`?`) with `NaN` and removed affected rows.
- Applied **One-Hot Encoding** for categorical variables.
- Scaled numerical features for uniformity.
- Encoded target variable: `0` for ≤50K and `1` for >50K income classes.

### ⚙️ Model Selection
- Algorithm: **K-Nearest Neighbors (KNN)**
- Train-Test Split: 70% training / 30% testing with stratification.
- Hyperparameter tuning: Optimal `K = 13` via cross-validation.

### 📏 Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score

---

## 📊 Results

### 📈 Model Performance
- **Accuracy**: 83.7%
- Good performance for low-income class; moderate performance on high-income due to class imbalance.

### 🔍 Key Insights
- Top correlated features:
  - `education_num`
  - `hours_per_week`
  - `capital_gain`

---

## 💬 Discussion

### Key Takeaways
- **Education** and **work hours** play a significant role in predicting income levels.
- Policy-makers can leverage such models to identify inequality drivers.
- Addressing class imbalance or using ensemble models may improve high-income prediction accuracy.

### Limitations
- Static census snapshot with no temporal or geographic variance.
- KNN doesn’t provide native feature importance; insights come from correlation analysis.

---

## ✅ Conclusion
The KNN-based classification model demonstrates strong performance in identifying income disparities using census attributes. While effective for low-income identification, further refinement with additional features or advanced algorithms can improve overall accuracy and fairness.

---

## 📚 References
- Brown, P., Smith, J., & Taylor, K. (2021). *Classification models in economic data analysis*. Journal of Applied Machine Learning, 34(2), 78–92.
- Jones, M., & Taylor, L. (2019). *Preprocessing techniques for categorical data*. Data Science Review, 27(3), 45–60.
- Lee, S. (2022). *Regional economics and income disparities*. Economic Studies Quarterly, 38(4), 120–139.
- Smith, R. (2020). *Exploring income inequality with machine learning*. Journal of Socioeconomic Studies, 12(1), 23–41.
- Taylor, K., & White, L. (2020). *Policy implications of income analysis*. Journal of Economic Policy, 40(1), 56–72.

---

## 🧠 Author  
**Mohammed Saif Wasay**  
*Data Analytics Graduate — Northeastern University*  
*Machine Learning Enthusiast | Passionate about turning data into insights*  

🔗 [Connect with me on LinkedIn](https://www.linkedin.com/in/mohammed-saif-wasay-4b3b64199/)

---

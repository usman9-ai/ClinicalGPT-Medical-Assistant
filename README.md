# Cyanotoxin Classification for Water Quality Control Using Transfer Learning

This project is an extension of existing research aimed at improving the classification of cyanotoxins for water quality control. By leveraging transfer learning, we built a model that classifies cyanotoxins into various categories, helping to ensure the safety of water systems. Our approach compares the performance of several machine learning classifiers and a fully connected neural network to identify the most accurate classification method.

## Overview
The primary goal of this research is to extend an existing cyanotoxin classification system by adding more classes and evaluating the effectiveness of different machine learning classifiers. The project builds on the *CyanotoxinZero* base model, which originally classified 10 cyanotoxin classes. We applied transfer learning techniques and experimented with a fully connected neural network (FCNN) and six additional machine learning classifiers.


## Dataset
The dataset comprises 13 cyanotoxins classes/genera, which is actually an extention of an existing public data set TCB-DS.

## Methodology
### Transfer Learning
We used transfer learning techniques on the CyanotoxinZero base model, which allowed us to leverage pre-trained layers while adding new fully connected layers for fine-tuning on the expanded dataset.

### Classifiers
We compared the FCNN model against six traditional machine learning classifiers:
- *Random Forest*
- *XGBoost*
- *KNN Classifier*
- *FM Classifier*
- *Gradient Boosting Classifier*
- *Gaussian Naive Bayes*

### Evaluation Metrics
The following evaluation criteria were used to compare the classifiers:
- *Macro Precision, Recall, and F1 Score*: Calculated without considering class imbalance.
- *Weighted Precision, Recall, and F1 Score*: Takes class imbalance into account by giving more weight to the larger classes.

### Results
Our findings show that the *Fully Connected Neural Network (FCNN)* outperformed all other machine learning classifiers in all the evaluation metrics, demonstrating superior classification performance. The FCNN provided the best scores in terms of macro and weighted precision, recall, and F1 scores.

## Installation

### Prerequisites
- Python 3.8+
- Ensure you have pip installed.

### Steps
1. *Clone the repository:*
   bash
   git clone https://github.com/your-username/CyanotoxinClassification.git
   cd CyanotoxinClassification
   

2. *Install required dependencies:*
   bash
   pip install -r requirements.txt
   

   


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

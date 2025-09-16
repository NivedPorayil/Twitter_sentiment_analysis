# Twitter Sentiment Analysis on Brands and Products  
**Author:** Nived P  
**Dataset:** Twitter Validation Dataset

---

## 📖 Project Overview
In today’s digital world, social media platforms have become spaces where people share their thoughts, reviews, frustrations, and praises about products, brands, and services. Understanding these sentiments can help businesses, marketers, and analysts make informed decisions and better engage with customers.

This project focuses on analyzing sentiments from tweets that mention various entities such as tech companies, video games, consumer brands, and more. By leveraging Natural Language Processing (NLP) techniques, the text data is cleaned and structured to reveal meaningful patterns.

The process involves transforming noisy, informal text into features that machine learning models can interpret. Techniques like tokenization, stemming, and stopword removal are applied to preprocess the tweets. The features are then extracted using TF-IDF vectorization, which helps in understanding the importance of each word within the dataset.

Multiple classifiers — including KNeighbors, Naive Bayes, Support Vector Classifier (SVC), Decision Tree, and Random Forest — are trained on the data to predict whether a tweet expresses a positive, negative, or neutral sentiment. The results highlight how different algorithms interpret the same data and demonstrate the power of machine learning in extracting insights from user-generated content.

This project not only builds a practical model but also showcases the importance of preprocessing and feature extraction in sentiment analysis workflows.

---

## 📂 Dataset
- Contains 1000 tweets with sentiment labels.
- Each entry includes:
  - `id`: unique identifier
  - `media`: mentioned entity (brand, product, or topic)
  - `target`: sentiment label (`Positive`, `Negative`, `Neutral`, or `Irrelevant`)
  - `text`: content of the tweet
- After cleaning, irrelevant entries are removed, and only tweets with clear sentiment labels are used for training and evaluation.

---

## ✅ Key Steps
1. **Data Loading & Exploration**  
   Inspecting the dataset structure, identifying missing values, and analyzing sentiment distribution.

2. **Data Cleaning**  
   Removing irrelevant rows, unnecessary columns, and mapping sentiment labels to numerical values.

3. **Text Preprocessing**  
   Using tokenization to split text, removing special characters and short words, applying stemming to normalize words, and eliminating stopwords to focus on meaningful content.

4. **Feature Extraction**  
   Using TF-IDF vectorization to convert text into numerical features that capture word importance across the dataset.

5. **Model Building**  
   Training different classifiers — KNeighbors, Naive Bayes, SVC, Decision Tree, and Random Forest — to predict sentiments.

6. **Model Evaluation**  
   Assessing the classifiers with confusion matrices, accuracy scores, and classification reports to understand performance.

---

## 📊 Results
The dataset showed a predominance of neutral sentiments, suggesting that many tweets expressed balanced or non-opinionated content. Among the classifiers, the **Support Vector Classifier (SVC)** delivered the most balanced performance across positive, negative, and neutral classes, making it a strong choice for sentiment classification tasks in noisy social media environments.

---

## 📂 Technologies & Libraries
- Python  
- Natural Language Processing (NLP)  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- NLTK  
- Scikit-learn

---

## 🚀 Usage
1. Clone this repository to your local environment:
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
2.Download the twitter_validation.csv dataset and place it in the repository directory.

3.Run the notebook or script to preprocess data, train models, and view evaluation results.

4.Experiment by adjusting preprocessing methods or testing new models to improve performance.

---

##🔮 Future Improvements
Expand Dataset: Incorporate more tweets or other social media data to improve model generalization.

Advanced Preprocessing: Apply lemmatization, spell correction, or deep learning embeddings like Word2Vec or BERT for richer text representations.

Hyperparameter Tuning: Use grid search or random search to find optimal parameters for classifiers.

Sentiment Intensity: Extend the model to classify sentiments on a scale (e.g., from highly negative to highly positive) rather than just three classes.

Real-time Analysis: Integrate the model with Twitter’s API to analyze live data and provide sentiment dashboards for user feedback monitoring.

---

##🤝 Contributing
Contributions are highly encouraged! Feel free to fork the repository, experiment with new models, refine preprocessing steps, or improve the evaluation metrics. Any feedback or pull requests are welcome.

---

##📜 License
This project is licensed under the MIT License. You are free to use, modify, and distribute it for educational and research purposes.

---

## ✍ Author

**NIVED P**  
*Aspiring Data Scientist | Machine Learning Enthusiast*

---
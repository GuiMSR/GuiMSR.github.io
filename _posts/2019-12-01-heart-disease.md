---

title: "Machine Learning Project: Heart Disease Prediction"
date: 2019-12-01
tags: [machine learning, logistic Regression, data science]
header:
  image: "/images/heart/heart_back.jpg"
excerpt: "Machine Learning, Heart Disease, Data Science"
mathjax: "true"
gallery:
  - url: /images/heart/hist1.png
    image_path: /images/heart/hist1.jpg
  - url: /images/heart/hist2.png
    image_path: /images/heart/hist2.jpg
  - url: /images/heart/hist3.png
    image_path: /images/heart/hist3.jpg
  - url: /images/heart/hist4.png
    image_path: /images/heart/hist4.jpg
  - url: /images/heart/hist5.png
    image_path: /images/heart/hist5.jpg
  - url: /images/heart/max_heart_achieved_age.jpg
    image_path: /images/heart/max_heart_achieved_age.jpg

gallery2:
  - url:  /images/heart/validation_curves.jpg
    image_path:  /images/heart/validation_curves.jpg
  - url: /images/heart/learning_curves.jpg
    image_path: /images/heart/learning_curves.jpg
---

#  1.Introduction

  The dataset comes from Driven Data ([Driven Data Website](https://drivendata.org)). The goal is to predict the probability of a patient having a heart disease/attack using measurements on patient health and cardiovascular statistics.  Data provided is a courtesy of the Cleveland Heart Disease Database via the UCI Machine Learning repository.

  This was a competition where I finished 617th on 3905 competitors. My final private score was 0.35499 using the log loss metric : `−(1/n)∑n(i)=1[y(i)log(y^i)+(1−y(i))log(1−y^i)]`. Pretty good for my first machine learning and data science competition !!

  The presented solution is not the one I submitted, but a better one after learning about feature engineering.

# 2. The Dataset and Visualization

  The dataset has 14 columns with different types and different scales. The first thing then it's to visualize the data and see if we can  already take out some information.

  Unfortunately, the different scatter plots and histograms don't give too much insight about the data. However, some of the plots were quite interesting.

  {% include gallery class="full" caption="Interesting plots and histograms." %}

  The distribution of these particular columns can be quite challenging for the model. The boxplots of those columns also help to reach that conclusion. Therefore, I decided to go with the log of those columns to have more "normal" distributions. The log of the column relative to the old peak depression wasn't helping the model so I let it be.


# 3. Model implementation
  I started by using the simple LogisticRegression algorithm which works quite well for this dataset (let's not overly complicate things). After splitting the data into train and test data, I tried to tweak the parameters until a was satisfied with the accuracy and the log loss score. However, I soon realized that the size of the test set had a major impact on the score (either the accuracy and the log loss), and decided to go with a test size of 0.05 (5% of the data). It is really small, and may cause some overfitting. But the original dataset is small, and therefore there is not much of a problem to sacrifice 5% of the data, just to have a small idea of how the model is predicting on unseen data.

  To do the hyperparameters tuning, I used the grid search with 5 cross-validation folds

  ````
  parameters = {'C': (np.linspace(0.001,120)), 'penalty':('l1', 'l2')}

  lgr = LogisticRegression()
  clf = GridSearchCV(lgr, parameters, cv=5)
  clf.fit(X_data, y_data.values.ravel())
  ````
And obtain `C=2.449959183673469` and the `l1` penalty with the **liblinear** solver. I chose this solver because it is the best one with a small dataset.

Afterwards I just trained the model and predicted the probabilities with `clf.predict_proba()` on the train and test set.

The final scores for the Logistic Regression are:
* train score: 0.847953216374269
* The test score: 0.8888888888888888
* The log loss value for the training: 0.3497573391237319
* The log loss value for the test: 0.36879487173164766

# 4. Validation and Learning curves

Here are the validation and learning curves:

{% include gallery id="gallery2" class="full" caption="Validation and learning curves." %}

# 5.Conclusion

This is a simple Machine Learning algorithm that I made to first learn how works a data science competition and then to have some fun while experimenting with the new tools I have learned.

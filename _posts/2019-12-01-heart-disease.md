---

title: "Machine Learning Project: Heart Disease Prediction"
date: 2019-12-01
tags: [machine learning, data science]
header:
  image: "/images/heart/heart_back.jpg"
excerpt: "Machine Learning, Heart Disease, Data Science"
mathjax: "true"

---

#  1.Introduction

  The dataset comes from Driven Data ([Driven Data Website](https://drivendata.org)). The goal is to predict the probability of a patient having a heart disease/attack using measurements on patient health and cardiovascular statistics.  Data provided is a courtesy of the Cleveland Heart Disease Database via the UCI Machine Learning repository.

  This was a competition where I finished 617th on 3905 competitors. My final private score was 0.35499 using the log loss metric : `−(1/n)∑n(i)=1[y(i)log(y^i)+(1−y(i))log(1−y^i)]`. Pretty good for my first machine learning and data science competition !!

  The presented solution is not the one I submitted, but a better one after learning about feature engineering.

# 2. The Dataset and Visualization

  The dataset has 14 columns with different types and different scales. The first thing then it's to visualize the data and see if we can   already take out some information.

  Unfortunately, the different scatter plots and histograms don't give too much insight about the data. However, some of the plots were quite interesting.

  The distribution of these particular columns can be quite challenging for the model. The boxplots of those columns also help to reach that conclusion. Therefore, I decided to go with the log of those columns to have more "normal" distribution. The log of the column relative to the old peak depression wasn't helping the model so I let it be.


# 3. Training and model selection
  A started by using the simple LogisticRegression algorithm which works quite well. After separating the data into train and test data, I tried to tweak the parameters until a was satisfied with the accuracy and the log loss score. However, I soon realized that the size of the test set had a major impact on the score (either the accuracy and the log loss).

  *C=*



Here's some basic text

And here's some *italics*

Here's some **bold** text

What about a [link](https://github.com/GuiMSR)

Here's a bullet list:
* item
1. item

Python code block:
```Python
  import numpy as np

  def test_function(x, y):
    z=np.sum(x,y)
    return z
```
Here's some inline code `x+y`

Here's some math: $$ 2*5 = 10 $$

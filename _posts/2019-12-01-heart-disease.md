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

  The dataset comes from Driven Data ([link](https://drivendata.org)). The goal is to predict the probability of a patient having a heart disease/attack using measurements on patient health and cardiovascular statistics.  Data is provided courtesy of the Cleveland Heart Disease Database via the UCI Machine Learning repository. Therefore, this was a competition where I finished 617th on 3905 competitors. My final private score was 0.35499 using the log loss metric : `−(1/n)∑n(i)=1[y(i)log(y^i)+(1−y(i))log(1−y^i)]`. Pretty good for my first machine learning and data science competition !!

  The presented solution is not the one I submitted, but a slightly better one.
#  2.The dataset

  The dataset has 14 columns such as:
    * slope_of_peak_exercise_st_segment
    * thal
    * resting_blood_pressure
    * chest_pain_type (4 types)
    * num_major_vessels
    * fasting_blood_sugar_gt_120_mg_per_dl
    * resting_ekg_results
    * serum_cholesterol_mg_per_dl
    * oldpeak_eq_st_depression
    * sex
    * age
    * max_heart_rate_achieved
    * exercise_induced_angina

    As you can see, there are a lot of features of different types and different scales. The first thing then it's to visualize the data and see if we can already take out some information.

# 3.Visualization

    Unfortunately, the different scatter plots and histograms didn't give too much insight about the data. However, some of the plots were quite interesting.

    <img src="{{ site.url }}{{ site.baseurl }}/assets/images/heart/max_heart_achieved_age.png" alt="Max heart rate achieved in function of age">
    As you may see, the younger the patient is, the higher is heart rate can go, which is normal.


    And here is some histograms to sense the data:
    <img src="{{ site.url }}{{ site.baseurl }}/assets/images/heart/hist1.png" alt="">
    <img src="{{ site.url }}{{ site.baseurl }}/assets/images/heart/hist2.png" alt="">
    <img src="{{ site.url }}{{ site.baseurl }}/assets/images/heart/hist3.png" alt="">
    <img src="{{ site.url }}{{ site.baseurl }}/assets/images/heart/hist4.png" alt="">
    <img src="{{ site.url }}{{ site.baseurl }}/assets/images/heart/hist5.png" alt="">

    As you can see, the distribution of those particular columns can be quite challenging for the model. The boxplots of those columns also help to reach that conclusion. Therefore, I decided to go with the log of those columns (resting_blood_pressure, serum_cholesterol_mg_per_dl and max_heart_rate_achieved) to have more "normal" distribution. The log of the column relatif to the old peak depression wasn't helping the model so I let it be.

# 4. Training and model selection



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

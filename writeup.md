# DAND-p4-identify-fraud-from-enron-email

### Enron Fraud Detection
--------------------------------------------------------------
Enron Submission Free-Response Questions


A critical part of machine learning is making sense of your analysis process and communicating it to others. The questions below will help us understand your decision-making process and allow us to give feedback on your project. 

Please answer each question; your answers should be about 1-2 paragraphs per question. If you find yourself writing much more than that, take a step back and see if you can simplify your response!


When your evaluator looks at your responses, he or she will use a specific list of rubric items to assess your answers. Here is the link to that rubric: [Link] Each question has one or more specific rubric items associated with it, so before you submit an answer, take a look at that part of the rubric. If your response does not meet expectations for all rubric points, you will be asked to revise and resubmit your project. Make sure that your responses are detailed enough that the evaluator will be able to understand the steps you took and your thought processes as you went through the data analysis.

Once you’ve submitted your responses, your coach will take a look and may ask a few more focused follow-up questions on one or more of your answers.  
We can’t wait to see what you’ve put together for this project!

Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

****************
Project Overview
****************

The goal of this project is to use our Enron dataset to create a Person of Interest (POI) identifier by combining Machine Learning techniques and publicly released financial data. We will be using numerical data from the dataset to determine if an individual should be classified as a person of interest. To score the efficacy of the machine learning models, I will score them against known POIs that were identified in the dataset. My investigative process will occur as follows:

Exploratory Data Analysis
Applying machine learning algorithms
Tuning and validating the machine learning algorithms

****************
Data Exploration
****************

In our data exploration, we uncover a number of features
    
    * 146 employees
    * 18 POIs
    * 128 non-POIs
    * 21 features per employee

*********************
Outlier Investigation
*********************

To find the outliers, I used a graph to see if any data points stood out, this way, we were able to visualize the data and zoom into data points that seem to not belong.Even though Ken Lay was one of the points that stood out, we kept it because he played a major role in the Enron scandal.

I found 2 outliers in my investigation

    * 'TOTAL' - this is not an individual but rather a summation of different numeric features and should be discarded

    * 'THE TRAVEL AGENCY IN THE PARK' - this is not an individual, but appears to be some sort of LLC entity used by the corporation for various (possibly illegal) business purposes. Since this is not an individual, I discarded it.

*********************
At least one new feature is implemented. Justification for that feature is provided in the written response. The effect of that feature on final algorithm performance is tested or its strength is compared to other features in feature selection. The student is not required to include their new feature in their final feature set.

Just a small change is required here: given the scores obtained by SelectKBest and decision trees, how good or bad the new features are?
To meet this criterion, we need a small paragraph comparing the scores obtained for the new features compared to those of the original ones, highlighting how good or bad the new features are.

*******************************************

With our analysis on the original feature we get

Accuracy: 0.775862068966
Precision:  0.142857142857
Recall:  0.125
Decision Tree algorithm run time:  0.005 s
Feature Importance Ranking: 
1 feature salary (0.337437907714)
2 feature from_poi_to_this_person (0.236734693878)
3 feature from_this_person_to_poi (0.109563164109)
4 feature to_messages (0.101688311688)
5 feature deferral_payments (0.0902902083254)
6 feature total_payments (0.0753246753247)
7 feature exercised_stock_options (0.048961038961)
8 feature bonus (0.0)
9 feature restricted_stock (0.0)
10 feature shared_receipt_with_poi (0.0)

vs.

Feature importance Ranking: 
1 feature salary (0.255056834004)
2 feature bonus (0.232918904311)
3 feature f_from_poi (0.140117302112)
4 feature f_to_poi (0.135416666667)
5 feature deferral_payments (0.0974911747407)
6 feature total_payments (0.0967261904762)
7 feature loan_advances (0.0322420634921)
8 feature restricted_stock_deferred (0.0100308641975)
9 feature deferred_income (0.0)
10 feature total_stock_value (0.0)
11 feature expenses (0.0)
12 feature exercised_stock_options (0.0)
13 feature long_term_incentive (0.0)
14 feature shared_receipt_with_poi (0.0)

Overall, with the new feature added, there has been a marginal difference
with the ranking of feature f_from_poi (0.140117302112),feature f_to_poi (0.135416666667) compared to the orginal features. In some ways, the new feature declined in importance ranking relative to the original feature of from_poi_to_this_person (0.236734693878),
from_this_person_to_poi (0.109563164109),
to_messages (0.101688311688).

What features did you end up using in your POI identifier,and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) 
 
 **************

Based on our dataset between the two set of features - financial(salary bonus, stock, etc) and communication(to and from emails). 

My assumption here is that communication between POI and POI would be more frequent than between POI and non-POIs. I am hoping to build an algorithm which would flag POIs among the population with good precision and recall score. I attempted to create two features, number of emails this person gets from POI(f_from_poi) and number of emails this person sends to POI(f_to_poi).

f_from_poi = number of emails this person gets from POI/total number of from messages

f_to_poi = number of emails this person sends to POI/total number sent messages

Feature Scaling
***************
Feature scaling puts the features on a more equal footing so that they can be compared in terms of relative impact. For instance if an important feature is extremely small relative to other features in the dataset, it may be overlooked unless it is rescaled to reveal its impact. We felt our project was sufficient to produce our desired results, however, if not we will revisit feature scaling. 


 In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]
*************

In our feature selection/engineering step,we selected and kept only impactful features in our dataset which contribute maximum information to the prediction variable. We do not want features which do not provide decent information. Having irrelevant features in our data decreases accuracy and quality of analysis. Univariate feature selection works by selecting the best features based on univariate statistical tests. SelectKBest removes all but the k highest scoring features. Here I used SelectKBest to see high scoring features.

Secondly, I used feature importance attribute of decision tree and this helped me to select features which would give me maximum precision and recall scores.

selectKBest scores ranking: 

1 feature salary (4.17424818808)
2 feature bonus (2.47526799584)
3 feature f_from_poi (1.14486649754)
4 feature f_to_poi (0.940272361115)
5 feature deferral_payments (0.337329786625)
6 feature total_payments (0.255505343107)
7 feature loan_advances (0.203646885336)
8 feature restricted_stock_deferred (0.198225231446)
9 feature deferred_income (0.193731697928)
10 feature total_stock_value (0.16482094858)
11 feature expenses (0.107627134646)
12 feature exercised_stock_options (0.0656280674818)
13 feature long_term_incentive (0.0546846481247)
14 feature shared_receipt_with_poi (0.0428473264338)
15 feature restricted_stock (0.0229248896842)
16 feature director_fees (0.00439829493019)

Feature importance Ranking: 

1 feature salary (0.255056834004)
2 feature bonus (0.232918904311)
3 feature f_from_poi (0.140117302112)
4 feature f_to_poi (0.111772486772)
5 feature deferral_payments (0.0967261904762)
6 feature total_payments (0.077380952381)
7 feature loan_advances (0.043754402254)
8 feature restricted_stock_deferred (0.0322420634921)
9 feature deferred_income (0.0100308641975)
10 feature total_stock_value (0.0)
11 feature expenses (0.0)
12 feature exercised_stock_options (0.0)
13 feature long_term_incentive (0.0)
14 feature shared_receipt_with_poi (0.0)


What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

For this exercise, I tried out several different algorithms to see which one would yield the best performance. They included GaussianNB, Adaptive Boosting (AdaBoost), DecisionTree.

NAIVE BAYES:

Naive Bayes recall score: 0.4
Naive Bayes precision score: 0.285714285714
accuracy: 0.8
NB algorithm time: 0.007 s

ADABOOST:

0.85
AB Recall Score: 0.2
AB Precision Score: 0.333333333333

DECISION TREE: 

Decision tree algorithm time: 0.002 s
accuracy: 0.875
recall score: 0.2
precision score: 0.333333333333

****************
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  

Tuning the parameters of the algorithm involves feeding in different parameters that will optimize the evaluation metrics (accuracy,precision score, etc.) If the parameters are not optimized, the algorithm will underfit or overfit itself to the data and reduce its performance
How did you tune the parameters of your particular algorithm? 

What parameters did you tune? 


If algorithm is not tuned well, it won't be trained well and it would not be able to make predictions on the unseen data. Here, to optimize its performance, parameters min_sample_split was varied manually and performance evaluated. We found algorithm performed well when min_sample_split was kept at 3.


(Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

******************

What is validation, and what’s a classic mistake you can make if you do it wrong? 

Validation in the context of Machine Learning is a type of verification method that tests the efficacy of the tuned algorithm. It involves splitting data into training and testing splits, using the training data to fit the model and then testing the efficacy with the test set. A classic mistake is to not split the data into train/test sets; if this is not done, the test results will return an overly optimistic result that does not truly represent the effectiveness of the model. The key is to remember that testing and training datasets must remain separate.


How did you validate your analysis?  

In my case, we used the k-fold cross validation technicque.

[relevant rubric items: “discuss validation”, “validation strategy”]

*********************

Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

The evaluation metrics that were given to me via Tester.py were: Accuracy, Precision, Recall, and F1 Score.

Precision: ~0.418 Precision is sometimes called the Sensitivity or Positive Predictive Value. Precision measures the number of POIs correctly classified by our algorithm against the total number of POIs identified by the algorithm, whether correctly or incorrectly identified (True Positives divided by True Positives + False Positives).

Recall: ~0.31550 Recall is also known as the sensitivity or True Positive Rate of a test. Recall measures the number of events the algorithm correctly classifies against the all correct events (True Positivies divided by True Positives + False Negatives). In this case, recall will be measuring the number of times an algorithm correctly identifies a POI versus the number of all correct POIs as given by the POI-key.

***********************

References
*************

Udacity Tutorial - https://classroom.udacity.com/nanodegrees/nd002/parts/0021345409/modules/317428862475460/lessons/2258728540/concepts/24032586940923

Udacity Forum - https://discussions.udacity.com/t/how-to-go-about-p5/259206/25

Udacity Mentor - GEORGE ZHONGYUE
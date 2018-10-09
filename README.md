# Multivariate Analysis of Non-parametric MANOVA and Mixed-Data PCA

​																	Authors: Xueji Wang, YumiaoLi





**Abstract**

In this project, we use PCA to extract the variables that could explain the most variations in the dataset. Outcome variable (house sale price) is divided into 4 quantile groups. Non-parametric MANOVA shows that there is significant difference in square feet (p = 6) between groups. Further, we build classifier including LDA, Decision Tree, Random Forest, SVM and multinomial logistic regression. Comparison of different methods shows that SVM perform best with average miss-classification rate equal to 0.1149.



Data source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data



## 1. Introduction

Residential real estate prices are fascinating. The homebuyer, the home-seller, the real estate agent, the economist, and the banker are all interested in housing prices. The goal of our project is to figure out what features could explain the most variation and whether they differed significantly between price groups. While this particular competition is no longer active, the premise proved to be a veritable playground for testing our knowledge of data cleaning, exploratory data analysis, statistics, and, most importantly, multivariate analysis learned in Multivariate Analysis class. What follows is the combined work of Xueji Wang and Yumiao Li. 



## 2.Data Description and Preprocessing

### 2.1 Data description

As stated on the Kaggle competition description page, the data includes 79 predictor variables (house attributes) and one target variable(price). Each of the predictor variables could fall under one of the following:

lot/land variables, location variables, age variables, basementvariables, roof variables, garage variables, kitchen variables, room/bath room variables, utilities variables, appearance variables, external features (pools,porches, etc.) variables. 

The training set on the website include response (sale price) value, and the test set only have features, leaving the response blank for participants to predict. For our project, we only use the training set, and through out the project our group wanted to mimic a real-world machine learning project as much as possible, which meant that although we were given both a training set and a test set, we opted to treat the given test set as if it were “future” data. As a result, we further split the Kaggle training data into a training set and a testing set, with an 75/25 split, respectively. This allowed us to evaluate models. 



### 2.2 Data Preprocessing

At first, the dataset contains 1460 observations and 79 features, among them 16 variables include missing values, so we delete the features with missing values directly. Then the dataset contains 63 features. Sale price is our goal to predict and we further divide sale price into 4 quantile groups. So we can use the predictors to perform PCA, use Sale price as response to do regression, and use Sale price group as response to conduct classification. 

In our experience, houses are often listed in terms of total number of bedrooms and total number of bathrooms. The original data had total number of bedrooms, but lacked total numberof bathrooms. So we add Total number of bathroom = FullBath + HalfBath + BasementHalfBath.

The following parts of our project are planned as below:

The third part is exploratory data analysis. EDA provide away to view and get first insight into the possible relationship between houseprice and key features. The fourth part is principal component analysis and MANOVA. PCA with both qualitative and quantitative features to capture the main variation in the data; We divide the sale price into groups and using non-parametric MANOVA to test whether there is significant different in house quality (include overall quality, kitchen quality, and exterior quality)between different house price groups. The fifth part conclude the findings in our project. 



## 3. Data Cleaning and EDA

Perform EDA to gain some insights from the data. Histogram shows the distribution of sale price and it can be observed that few houses are more than $400,000. 95% confidence interval of sale price is (25087.48, 336754.9).

From Boxplots (Continuous variable v.s categorical variable) and Scatterplot (both numeric variables), House price differs between Neighborhood, house quality, number of bedrooms. aslo, there is positive relationship between house price and living area.



## 4. Multivariate Analysis

### 4.1 Mixed data PCA

Every house has 63 features, however, not all of them could explain the variation within them, for example, the utility feature may notdiffer from house to house. On the contrary, the ground living area, or theNeighborhood could be varied considerably. Thus, we perform dimension reduction to find the variable that could explain the most variation within the data.

In order to perform dimension reduction to the explanatory variables, and to understand what features could contribute to the variation in the original data most R package PCAmixdata extends standard multivariate analysis methods to incorporate this type of data. The key techniques included in the package are principal component analysis for mixed data. We used PCAmixdata package and PCAmix() function to find the principal component and variables that contributeto the component most. We select first 5 PCs by scree plot.

![scree plot](/Users/xuejiwang/Documents/3_My_UNC_Charlotte/2018Spring/STAT_7133/Project/Charts/ScreePlot.png)





We sort the variables by loading, and found the followingvariables contribute the most to the 1st PC: Neighborhood, OverallQuality, construction year, Exterior quality, Kitchen Quality, Size of garagein car capacity, Foundation, Size of garage in square feet, Remodel date, Totalbasement square feet, The first principal component can be interpreted as **Entire Quality index**.

In the 2nd dimension: Neighborhood, Style of dwelling, Secondfloor square feet, Foundation, Exterior covering on house more than 2 type,Exterior covering on house first type, General zoning classification of thesale, Basement finished square feet, Total rooms above grade, Paved driveway. Thesecond principal component can be taken as **style and exterior inde**.

The 3rd dimension: Neighborhood, Above groundliving area, Total rooms above grade, Exterior covering on house first type,Exterior covering on house more than 2 type, Type of dwelling, Number ofbedrooms, Lot Area, First floor square feet.The third PC describe the variation mainly in **space**.

From PCA, we basically know that Neighborhood, House Quality and the living space mainly decide the house price.



### 4.2 Non-Parametric Manova

Unlike in classical multivariate analysis of variance, multivariate normality is not required for
the data. In our house price model, there is no normality assumption. We are interested in different house square feet have effect on the discriminant of house price. 

Results show that the difference between the three multivariate distributions is highly significant



### 4.3 Classification

To classify which price group (1,2,3,4) one house will belongs to. We performclassification and train the classifier on training using multinomial logisticregression, decision tree and random forest and compare the misclassification rate.



4.3.1 LDA

In linear discriminant analysis, we assume the dataset ismultivariate normal, and we classify one observation to the population forwhich pif(x|pi)is largest. Decision rule is based on the linear score function, a function ofthe population means for each of g population, as well as the pooledvariance-covariance matrix. Classify the sample unit into the population thathas the largest estimated linear score function.

4.3.2 Tree classification

Classification tree mainly include the idea of recursivepartitioning of the space of the independent variables and prune the tree usingvalidation data. 

For tree classification, Neighborhood, Overall Quality, Constructiondate, Foundation, Above ground living area, Exterior quality, Total square feetof basement, Kitchen Quality, First floor square feet, Size of garage in squarefeet, Size of garage in car capacity, Second floor square feet, Total roomsabove ground, Total number of bathroom, Bedroom above ground, Basement finishedsquare feet, Lot Area, Style of dwelling, Condition1, General zoningclassification of the sale contribute to the sale price the most. The totalmisclassification rate on testing set is 0.369863

4.3.3Random Forest

In random forest, we build decision trees, each time a splitin a tree is considered as a random sample of several predictors is taken at each split, and typically we choose  the number of predictors considered at eachsplit is approximately equal to the square root of the total number ofpredictors.

For random forest classifier, we sort by the importance of variable and get the result, that Above ground living area, Overall Quality, Neighborhood, Total square feet of basement, First floor square feet, Size of garage in square feet, Construction date, Lot Area, Remodel date, Basement finished square feet, Size of garage in car capacity, Exterior quality, Total number of bathroom, Second floor square feet, Kitchen Quality, Exterior covering on house more than 2 type, , Exterior covering on house first type, Total rooms above ground, could explain the sale price the most.The total misclassification rate on testing set is 0.22.

4.3.4 Support Vector Machine

Support vector machine maximize the margin around theseparating hyperplane, and the decision function is fully specified by asubsect of support vectors which are the data points that lies closest to thedecision surface.

4.3.5 Multinomial LogisticRegression

Multinomial logisticregression is used to model nominal outcome variables, in which the log odds ofthe outcomes are modeled as a linear combination of the predictor variables.Wald-Z test shows that the following variables has p-value larger than 0.05.Which means that they do not contribute to the price group variation. Themisclassification rate of multinomial logistic regression is 0.2802198.

4.3.6 Comparison 

Repeat the process for 1000 times, and compare the misclassificationrate between 5 methods

![comparison of classifiers](/Users/xuejiwang/Documents/3_My_UNC_Charlotte/2018Spring/STAT_7133/Project/Charts/ComparisonBetweenMethods.png)

## 5. Conclusion

Neighborhood, Overall Quality, Construction date, Exterior Quality, Kitchen Quality, Garage Area, Remodel date, Total basement square feet, above ground living area, first floor square feet, Sale Condition, Sale Type can explain the most variations in the dataset. MANOVA shows that the square feet measurements (p=6) differed significantly between price groups (g = 4). Comparison between different classification methods shows that random forest and support vector machine perform better than the others. SVM is a powerful technique and especially useful for data whose distribution is unknown (also known as non-regularity in data). 



## Reference:

[1]Davis, A. W. ‘Asymptotictheory for principal component analysis: Non-normal case’, Australian Journal of Statistics 19, 206–212.

[2]Fanyin He, Sati Mazumdar,Gong Tang, Triptish Bhatia, Stewart J. Anderson, Mary Amanda Dewa, RobertKrafty, Vishwajit Nimgaonkara, Smita Deshpande, Martica Hall, and Charles F.Reynolds III. ‘Non-parametric MANOVA approaches for non-normal multivariate outcomes with missing values’, Communicationin statistics-theory and methods. 2017,Vol.46,No.14, 7188-7200.

[3]Marielle.Linting and Anita Van Der Kooij. ‘Nonlinear Principal Components Analysis with CATPAC: A Tutorial’, Journal of Personality Assessment, 94(1),12-25,2012.

[4]Niitsuma, Hirotaka; Okada, Takashi. ‘Covariance and PCA forCategorical Variables’, <https://arxiv.org/abs/0711.4452>

[5]Stanislav Kolenikov, Gustavo Ángeles. ‘The Use of Discrete Data inPCA: Theory, Simulations, and Applications to Socioeconomic Indices’, TechnicalReport, MEASURE/Evaluation project, Carolina Population Center, University of North Carolina, Chapel Hil


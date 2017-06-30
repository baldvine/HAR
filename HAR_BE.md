# Human Activity Recognition
Baldvin Einarsson  
6/30/2017  



Human Activity Recognition (HAR) is an interesting topic, with monitors and sensors becoming widely used. See information on original data in the following website:
[http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har)

Here, we look at the weight lifting exercise dataset. This dataset is concerned with classifying whether an a weight lifting exercise was done properly.

# Import Data

Let's download training and test sets from the following websites:

* The data is the following (which will be split into training/test for cross-validation):
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

* The data to predict on:
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

Let's import these data files into `R`, into objects `mainData` and `data2predict`. After a little bit of trial and error, we settle on `na.strings = c("NA","#DIV/0!")` in order to handle missing or invalid values:

```r
mainData <- read.csv(file = "./data/pml-training.csv",
                     header = TRUE, stringsAsFactors = FALSE, 
                     na.strings = c("NA","#DIV/0!"))

data2predict <- read.csv(file = "./data/pml-testing.csv",
                     header = TRUE, stringsAsFactors = FALSE,
                     na.strings = c("NA","#DIV/0!"))
```

The data contains 160 columns. The columns are the same except for two columns, one from each dataset. The `mainData` contains the column "classe" and `data2predict` contains the column "problem_id".

Six people performed 10 repetitions of the "Unilateral" Dumbbell Biceps Curl", in five different ways. The variable we're interested is called `classe`, and the unique values labeled as A through E, with the following descriptions:

The classes stand for the following:

Class   Description                            
------  ---------------------------------------
A       Exactly according to the specification 
B       Throwing the elbows to the front       
C       Lifting the dumbbell only halfway      
D       Lowering the dumbbell only halfway     
E       Throwing the hips to the front         

What now follows is some menial data handling, removing columns which do not contain any relevant information etc. Please feel free to skip to the [section on splitting data](#splitData) or [model building section](#modelBuilding), where the real action is.

## Remove unwanted columns

We see that the first seven columns are of no use, as they contain line numbers, name of person, time stamps and two columns on "windows". We remove these columns from the `mainData`


```r
mainData <- 
    mainData[,-match(c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2",
                         "cvtd_timestamp","new_window","num_window"),
                     names(mainData))]
```

We also notice that some columns contain only NA values, or a significant amount of NAs. Let's inspect further:

```r
countNAs <- sapply(mainData, function(x){sum(is.na(x))})
```
The columns with no NAs are the actual measurements, and they have the following prefixes:

```r
regexpr("^[A-Za-z]+", names(countNAs[countNAs == 0])) %>% 
    regmatches(names(countNAs[countNAs == 0]), m = .) %>% 
    unique()
```

```
[1] "roll"   "pitch"  "yaw"    "total"  "gyros"  "accel"  "magnet" "classe"
```
The minimum number of NAs in those columns which do have NA values, is 19216 (out of 19622). These columns are the derived quantities such as max, min, average, variance etc. In fact, all these NA columns have the following prefixes:

```r
regexpr("^[A-Za-z]+", names(countNAs[countNAs > 0])) %>% 
    regmatches(names(countNAs[countNAs > 0]), m = .) %>% 
    unique()
```

```
[1] "kurtosis"  "skewness"  "max"       "min"       "amplitude" "var"      
[7] "avg"       "stddev"   
```
Let's remove these columns with derived valued and a bunch of NAs:

```r
mainData <- mainData[,names(countNAs[countNAs == 0])]
```


Ok, so now all the columns apart the last one (the classifier), are numerical values:

```r
sapply(mainData[,-length(mainData)], class) %>% unique
```

```
[1] "numeric" "integer"
```


This brings us down to 52 predictors, all numeric inputs. From now on we expect any predictive algorithm to be able to find important columns, and we stop cleaning the data.

## Split into training and test set {#splitData}

For cross-validation, let's split the data into a training and test set randomly with 60% and 40% of the initial data each, respectively.


```r
set.seed(42)
inTrain <- createDataPartition(mainData$classe, p = 0.6)[[1]]
training <- mainData[inTrain,]
test <- mainData[-inTrain,]
```

# Building predictive models {#modelBuilding}

Here, it is easy to go crazy and have lots of fun with various classification models. Let's contain ourselves and only try a few for now. We'll create a list of model fits, predictions and confusion matrices, whose element names correspond to the model used.






First, let's try linear discriminant analysis, LDA:

```r
myMethod <- "lda"
modelFit[[myMethod]] <- 
  train(classe~., method = myMethod, data = training, na.action = na.omit)

# Predict on both training and test set:
training.pred[[myMethod]] <- predict(object = modelFit[[myMethod]], newdata = training)
test.pred[[myMethod]] <- predict(object = modelFit[[myMethod]], newdata = test)

# Obtain confusion matrices for both sets:
confMat.training[[myMethod]] <- confusionMatrix(training.pred[[myMethod]], training$classe)
confMat.test[[myMethod]] <- confusionMatrix(test.pred[[myMethod]], test$classe)
```

Let's obtain the predictions and confusion matrices for gradient boosting (method "gbm") and random forests (method "rf"). Since the change from the above chunk only involves changing the method name and minor changes, we don't show the code. Note that the code is available in the `.Rmd` file.









Note that since gradient boosting and random forests are usually very computationally heavy, we also preprocess with principal components (with label having suffix ".wPCA"), explaining 90.0 percent of the variance. It is also interesting to compare the in and out of sample errors with or without preprocessing.

## Inspecting accuracy and in and out of sample errors

Let's now do some basic check on how well the metods above performed. To this end, we use the aptly named object "Accuracy", in the `overall` element of the confusion matrices. For all methods used we inspect the accuracy on both the training and the test set:


Method      Accuracy on training set   Accuracy on test set
---------  -------------------------  ---------------------
lda                            0.704                  0.701
gbm                            0.975                  0.963
gbm.wPCA                       0.843                  0.807
rf                             1.000                  0.993
rf.wPCA                        1.000                  0.968

It is interesting to note that both models using random forests have perfect predictions on the training set. However good that may sound, it smells like overfitting. Our goal should not be a perfect fit on the training set, but rather obtain some kind of balance between variance and bias of our model fit.


For that reason, I would prefer gradient boosting (method "gbm"). Let's visualize its confusion matrix using plotly (*the R-markdown has code for those interested*). **Unfortunately, the `.md` file does not support these interactive plots. See `.html` file.**
<!--html_preserve--><div id="htmlwidget-ecdcf905329490dcf0cf" style="width:672px;height:480px;" class="plotly html-widget"></div>
<script type="application/json" data-for="htmlwidget-ecdcf905329490dcf0cf">{"x":{"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"title":"Confusion matrix for gradient boosting","xaxis":{"domain":[0,1],"title":"Reference","type":"category","categoryorder":"array","categoryarray":["A","B","C","D","E"]},"yaxis":{"domain":[0,1],"title":"Prediction","type":"category","categoryorder":"array","categoryarray":["A","B","C","D","E"]}},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"modeBarButtonsToRemove":["sendDataToCloud"]},"data":[{"colorbar":{"title":"","ticklen":2},"colorscale":[[0,"#F7FBFF"],[0.000670476849623938,"#F7FBFF"],[0.000743994048047616,"#F7FBFF"],[0.00268190739849575,"#F6FBFF"],[0.0027359781121751,"#F6FBFF"],[0.00319197446420429,"#F6FBFF"],[0.00712290413651807,"#F6FAFF"],[0.00791433792946453,"#F5FAFE"],[0.00846976858027022,"#F5FAFE"],[0.0112930247736936,"#F5FAFE"],[0.0118559051527588,"#F5F9FE"],[0.0119039047687618,"#F5F9FE"],[0.0162337231121846,"#F4F9FE"],[0.0254781202857097,"#F2F8FD"],[0.0260397916816665,"#F2F8FD"],[0.0288305045338293,"#F1F7FD"],[0.0356145206825904,"#F0F6FD"],[0.960122848661479,"#093A7A"],[0.967132094980565,"#093878"],[0.979096167230662,"#083573"],[0.98178734126299,"#083572"],[1,"#08306B"]],"showscale":false,"x":["A","A","A","A","A","B","B","B","B","B","C","C","C","C","C","D","D","D","D","D","E","E","E","E","E"],"y":["A","B","C","D","E","A","B","C","D","E","A","B","C","D","E","A","B","C","D","E","A","B","C","D","E"],"z":[0.991224940021401,0.107929354300456,0.0560017920860206,0.0518475847365213,0,0.168305513035224,0.971260268189686,0.158218008066137,0.0513327002339345,0.0256663501169673,0,0.159952478322899,0.980810024466291,0.108147614087175,0.0270369035217938,0,0.0881819129227673,0.187062085817071,0.9747990980161,0.0836567079799741,0,0.0912237650618882,0.105336130629945,0.126293583954719,0.982157041884706],"hoverinfo":"text","text":["Prediction:  A <br>Reference:  A <br>Percentage predicted:  98.3 %","Prediction:  B <br>Reference:  A <br>Percentage predicted:  1.2 %","Prediction:  C <br>Reference:  A <br>Percentage predicted:  0.3 %","Prediction:  D <br>Reference:  A <br>Percentage predicted:  0.3 %","Prediction:  E <br>Reference:  A <br>Percentage predicted:  0.0 %","Prediction:  A <br>Reference:  B <br>Percentage predicted:  2.8 %","Prediction:  B <br>Reference:  B <br>Percentage predicted:  94.3 %","Prediction:  C <br>Reference:  B <br>Percentage predicted:  2.5 %","Prediction:  D <br>Reference:  B <br>Percentage predicted:  0.3 %","Prediction:  E <br>Reference:  B <br>Percentage predicted:  0.1 %","Prediction:  A <br>Reference:  C <br>Percentage predicted:  0.0 %","Prediction:  B <br>Reference:  C <br>Percentage predicted:  2.6 %","Prediction:  C <br>Reference:  C <br>Percentage predicted:  96.2 %","Prediction:  D <br>Reference:  C <br>Percentage predicted:  1.2 %","Prediction:  E <br>Reference:  C <br>Percentage predicted:  0.1 %","Prediction:  A <br>Reference:  D <br>Percentage predicted:  0.0 %","Prediction:  B <br>Reference:  D <br>Percentage predicted:  0.8 %","Prediction:  C <br>Reference:  D <br>Percentage predicted:  3.5 %","Prediction:  D <br>Reference:  D <br>Percentage predicted:  95.0 %","Prediction:  E <br>Reference:  D <br>Percentage predicted:  0.7 %","Prediction:  A <br>Reference:  E <br>Percentage predicted:  0.0 %","Prediction:  B <br>Reference:  E <br>Percentage predicted:  0.8 %","Prediction:  C <br>Reference:  E <br>Percentage predicted:  1.1 %","Prediction:  D <br>Reference:  E <br>Percentage predicted:  1.6 %","Prediction:  E <br>Reference:  E <br>Percentage predicted:  96.5 %"],"type":"heatmap","xaxis":"x","yaxis":"y"}],"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":[]}</script><!--/html_preserve-->



# Prediction

Finally, we predict the values on the prediction data set, `data2predict`, using the gbm model fit above. The data contains 20 rows, and so the output is a vector of length 20:

```r
print(selectedMethod)  # Value was set above
```

```
[1] "gbm"
```

```r
predict(object = modelFit[[selectedMethod]], newdata = data2predict)
```

```
 [1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E
```



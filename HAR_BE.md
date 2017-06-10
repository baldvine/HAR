# Human Activity Recognition
Baldvin Einarsson  
6/3/2017  



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

The data contains 160 columns. The columns are the same except for two columns, one from each dataset. The `mainData` contains the column "classe" and "`data2predict` contains the column problem_id".

Six people performed 10 repetitions of the "Unilateral" Dubbell Biceps Curl", in five different ways. The variable we're interested is called `classe`, and the unique values labeled as A through E, with the following descriptions:

The classes stand for the following:

Class   Description                            
------  ---------------------------------------
A       Exactly according to the specification 
B       Throwing the elbows to the front       
C       Lifting the dumbbell only halfway      
D       Lowering the dumbbell only halfway     
E       Throwing the hips to the front         

## Remove unwanted columns

We see that the first seven columns are of no use, as they contain line numbers, name of person, time stamps and two columns on "windows". We remove these columns from the `mainData`


```r
mainData <- 
    mainData[,-match(c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2",
                         "cvtd_timestamp","new_window","num_window"),
                     names(mainData))]
```

We also notice that some columns contain only NA values. Let's find those columns for removal:


```r
colsNA <- 
    sapply(names(mainData), 
        function(colName){sum(is.na(mainData[,colName]))==nrow(mainData)})
names(colsNA)[colsNA]
```

```
[1] "kurtosis_yaw_belt"     "skewness_yaw_belt"     "kurtosis_yaw_dumbbell"
[4] "skewness_yaw_dumbbell" "kurtosis_yaw_forearm"  "skewness_yaw_forearm" 
```

```r
mainData <- mainData[,!colsNA]
```

Ok, so now all the columns apart the last one (the classifier), are numerical values:

```r
sapply(mainData[,-length(mainData)], class) %>% unique
```

```
[1] "numeric" "integer"
```


## Split into training and test set

For cross-validation, let's split the data into a training and test set randomly with 60% and 40% of the initial data each, respectively.


```r
set.seed(42)
inTrain <- createDataPartition(mainData$classe, p = 0.6)[[1]]
training <- mainData[inTrain,]
test <- mainData[-inTrain,]
```

# Building predictive models



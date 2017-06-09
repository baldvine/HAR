# Human Activity Recognition
Baldvin Einarsson  
6/3/2017  



Human Activity Recognition (HAR) is an interesting topic, with monitors and sensors becoming widely used. See information on original data in the following website:
[http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har)

Here, we look at the weight lifting exercise dataset. This dataset is concerned with classifying whether an a weight lifting exercise was done properly.

# Data

Let's download training and test sets from the following websites:

* The training data:
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

* The test data:
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

Let's import these data files into `R`, into objects `training` and `test`.


The data contains 160 columns. Six people performed 10 repetitions of the "Unilateral" Dubbell Biceps Curl", in five different ways. The variable we're interested is called `classe`, and the unique values labeled as A through E, with the following descriptions:

The classes stand for the following:

Class   Description                            
------  ---------------------------------------
A       Exactly according to the specification 
B       Throwing the elbows to the front       
C       Lifting the dumbbell only halfway      
D       Lowering the dumbbell only halfway     
E       Throwing the hips to the front         


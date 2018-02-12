# Questions

## How would you define Machine learning?

People tends to say that "Machine learning is a field of
computer science that gives computers the ability to learn
without being explicitly programmed". Which is a definition
which is clear to anyone that has ever used machine
learning, but gargabe to anyone that has no idea what ML is.
If it was me, I'd define machine learning as the set of
techniques aimed at giving meanings to data. Maybe what's
really different from other ways of give meanings to data is
the fact that a model is trained againsts data, instead of a
human being trained against data and trying to formulate
some heuristics.

## Can you name four types of problems where it shines?

Find all the cats in the picture; finds out what this
person did write on paper; recognize if this music is from
this artist; find out how best Go players play.

## What is labeled training set?

You have (sample, label) pairs in a labeled training set.
You want to use those pairs to understand the relationship
between _sample_ and _label_ and (maybe, as in supervised
learning) you want to be able to predict the label for a new
value.

## What are the two most common supervised task?

Classification and regression.

## Can you name four common unsupervised tasks?

Clustering, features extraction (i.e. identify relationships
between features such that you can use only one of them, as
the others are only functions of the first), anomaly
detection. Also, _visualization_ algorithms maybe. But still
not getting it.

## What type of Machine Learning algorithm would you use to
## allow a robot to walk in various unknown terrains?

Some supervised learning technique. A model is trained on
labeled data collected on known terrain, then the model is
used to make prediction on the unknown terrain.

## What type of ML algorithm would you use to segment your
## customers into multiple groups?

Some clustering algorithm.

## Would you frame the problem of spam detection as a
## supervised learning problem or an unsupervised learning
## problem?

A simple solution to spam detection would be to collect
spam, collect ham, train the model, predict label for
incoming mails. That would make the problem of spam
detection a supervised learning problem. Problem might be
you may need to collect _a lot_ of spam to get good
performance.

Potentially another way of solving this problem would be to
run a clustering algorithm, hopefully identifying few
clusters. Then let the user mark which cluster is spam and
which ham. Then classify incoming mails based on which
cluster it owns. Is it one marked by a human as spam? Then
it's spam.

## What is an online learning system?

Opposed to a batch learning system, an online learning
system is trained by a stream of data and it's continuously
updated. Think of anomaly detection: you want to be able to
find a transaction which can be a fraud immediately, not
after datas has been collected.

## What is out-of-core learning?

ML techniques that use a set/cluster of machines because
data is distributed. Maybe.

## What type of learning algorithm relies on a similarity
## measure to make prediction?

Instance-based learning. To make a prediction on a new value
a distance from the previous instances is computed.

## What is the difference between a model parameter and a
## learning algorithm's hyperparameter?

It's a parameter of the learning algorithm _not_ the model.
For example, how conservative to previous data it must be.

## What do model-based learning algorithms search for? What
## is the most common strategy they use to succeed? How do they
## make predictions?

Model-based learning algorithms search for a _model_. ???.
???.

## Can you name four of the main challenges in ML?

Overfitting: you are too focused on current data that you
can't predict the future. Data are not enough. Data are of
low quality (noise). Underfitting, e.g. assuming a
relationship is linear while it's a lot more complex.

## If your model performs great on the training data but
## generalizes poorly to new instances, what is happening? Can
## you name three possible solutions?

That's called _over_fitting. Possible solutions: 1, use a
simpler model: any scatter plot can be thought of a
super-complex function; in reality it might be a linear
function + noise. 2, get more training data: _maybe_ this
way by using more points algorithm is more likely to make
the simpler model to emerge (still confused about that). 3,
reduce the noise, such to confuse less the algorithm.

## What is a test set and why would you want to use it?

A test set is a portion of the sample _not_ used during
training. You use it to immediately measure the performance
of the model.

## What is the purpose of a validation set?

Validation set is used while training. It's a portion of the
data used to reduce the risk of overfitting. The idea is to
use validation set to check than any increase over the
training set corresponds to an increase of performance on
the validation set too - which has not used by the training
algorithm.

## What can go wrong if you tune hyperparameters using the
## test set?

Maybe overfitting and not realizing it? Maybe!

## What is cross-validation and why would you prefer it to a
## validation set?

Cross-validation is a technique to measure performance of
the model. To my understanding, the idea of cross-validation
is to _randomize_ the choice of which part of the dataset is
for training and which part is for test. That may be very
useful in cases in which the dataset is small and taking out
from the training part of the sample might be too much.

vim: set tw=60:

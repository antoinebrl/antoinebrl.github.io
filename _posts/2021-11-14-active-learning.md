---
layout: post
title: Why Active&nbsp;Learning is not widely adopted?
date: 2021-09-26 11:12:00 +0100
toc: true
description: "I recently spent quite some time learning and playing with Active Learning.
The claim of active learning is quite appealing: a model can achieve the same, or even better,
performances with fewer annotated samples if it can select which specific instances need to
be labeled. Sounds promising, right? In this article,
I will attempt to reflect on the promises of active learning to show that the picture is
not all bright. Active Learning has a few drawbacks, however, it represents a viable solution
when building datasets at scale or when tackling the diversity edge cases."
---

##  The Promises of Active Learning

Aside from high amounts of computations, deep learning models need a crazy amount of
**labeled data** to learn from. Collecting data is often not a problem as
**data is usually very abundant**. However, **generating the labels is the bottleneck**.
This process is very tedious, expensive, and error-prone.

By going back and forth between the annotation and training phase, active learning tries to
**identify the most valuable data point for the task**. The model is first trained on a small
number of labeled samples. Then, it selects the unlabeled data for which
the prediction is unsure. We say that the model formulates a query of unlabeled data.
When the annotators provide the labels (a.k.a. the answers) the model
can be retrained on a larger dataset. The process is repeated several times until the
annotation budget expires or performances are deemed good enough. By doing so,
fewer samples are required to reach the same, or higher level of performance.
This should translate to **faster and cheaper annotations**.

It should look something like this:

<figure>
  <img src="/assets/img/active-learning-illustrate.svg" alt="Benefit of active learning illustrated">
</figure>

The rest of the article does not assume any more knowledge about active learning than
the desctiption above. If you are interested in learning more about the different
scenarios and query strategies, I recommend reading
["Active Learning Literature Survey" by Burr Settles](http://burrsettles.com/pub/settles.activelearning.pdf)
or the [Wikipedia article on this topic](https://en.wikipedia.org/wiki/Active_learning_(machine_learning))

## What's wrong with Active Learning?

You can find a lot of content which are drawing a very flattering picture of active
learning. By sharing my views, I would like to share some reasons why it's not so
widely embraced.

### Static dataset is the rule

All publicly available datasets are inert. The second they become public, they
freeze: no new sample or annotation will be added or modified. This happens
for good a reason: **having a static dataset ensures a fair comparison of methods
and algorithms**. It leads to a sharp separation between building the datasets and training
a model.

However, in the industry, the story is completely different. **Data keeps on being generated
and labeling is often a continuous effort**. As more and more code are shared
under permissive open source licenses, the real competitive advantage comes from the size
and the quality of the datasets. I saw some large fully annotated datasets being sold for a few
million dollars. (This dataset was then copied into a big hard drive and sent over by mail
to the other side of the Earth to complete the exchange &#128552;.
But, that is a story for another time!).

The curation process is often overlooked. For sure it's not the sexiest part of
the job, but I believe people should talk more about it because it's a key aspect
of the success of any ML project. If I were to guesstimate, I would say for one
paper on dataset curation and creation, there are more than ten thousand papers claiming
state-of-the-art for their models.


### Training is easy, Evaluation is hard

Training a model is great fun. Unfortunately, that's the easy part!
**The real struggle is to perform a fair and challenging evaluation of the predictive
capabilities of a model**.
It is now considered the golden rule to set aside a portion of the labeled samples
before doing any training. The goal is to understand if the model behaves
well on unseen data. Reporting performances on the training set falls short of
the desired assessment.

Given a dataset with annotations (and some rich metadata), it becomes easier to build a
representative and challenging test set before training. But, as **active learning combines
the annotation and the training phases, it is unclear how and when the test set is built**.
This is often the major flaw in all publications on active learning techniques:
they rely on already well-made test sets.

If you can afford to select random samples for evaluation then you are good to go.
But, if you need stratified sampling or more challenging splits, then active
learning might not be suitable. A practical way is to initiate the test set with random samples
and to enlarge or rebuild it with samples from the training set when it reaches a certain size.
The modification of the test set can be done several times over the course of the training.
However, changing the test set too often makes it hard to track learning progress.


### There is no such thing as a free lunch

As the adage says, nothing comes for free. **Like any machine learning
technique, active learning requires a good amount of tuning to gain maximum performances**.

The final training set should contain samples which are **representative of
the latent distribution, as diverse as possible and as informative as possible
to solve the task**.
An active learning algorithm needs to carefully balance these three elements.
Indeed, by focusing solely on the most representative samples, we are likely
to draw very similar items from high density regions. Some diversity needs to
be enforced to cover more of the input space.
However, if we optimize for diversity, we end up with outliers or items on the
outskirt of the domain. **A good exploration strategy is often dataset specific,
there is no clear winner**.

Finally, the **pertinence of a sample for the task is often related to the
training dynamic of a specific model**. A data point might be selected because it lies
on the decision boundary of a model (high uncertainty).
But, a different model would have placed this boundary somewhere else and queried
different data points. Even worse, an item might be considered very
informative in the early step of the process but it might become irrelevant when new
annotations get added. **This greedy strategy is therefore dependent on
an architecture, a given initialization and a set of hyper-parameters**.


### The rise of other training techniques

Another assumption encountered in the literature is that the training
is fully supervised. Despite being the most widely used training paradigm, supervised
learning is no longer the only technique to achieve good results.

The recent surge of research in unsupervised, semi-supervised, and self-supervised learning
leads to a rapidly decreasing performance gap with supervised techniques. These approaches
leverage huge amounts of unlabeled data; the more we feed them the better they get.
**Labeling all of the samples might be impossible but not utilizing the unlabeled
ones is a waste**.


### The low return on investment

Active Learning was invented to reduce the amount of required labeled samples.
Here, the number of annotations serves as a proxy to cut costs and time.
Yet, **not all samples are created equal**. Some data points might take longer
to fully annotate or they might require an expert eye, which is often scarce and/or expensive.
If that is the case, the dataset breaks the assumption of constant cost per sample
which might not yield the expected gain.

In addition, the benefits are minimized because active learning requires a lot
of computations. Every time a new batch of data gets annotated a new training
is started. This successive retraining on ever-growing datasets tends to increase drag costs.
If not considered upfront, **the computational costs can eat all the savings
or make it more expensive than traditional annotation methods**. These computational
costs are not reported in academic publications.

On top of this, time is no longer a constraint. **The labeling industry
scale vertically really well** by hiring thousands of workers all around the world.
Suppose you retrain your model every 24h, querying enough data for a full day of
work for hundreds of annotators represents a challenge for any active learning
methods. As batches get bigger they bring less and less information per sample.

Finally, **most of the costs induced by active learning are from implementation,
experiment tracking, version control, automation, maintenance, etc**.
This is referred to as MLOps and should come as no surprise as active learning
is a branch of machine learning. These operations are done
by skilled and highly paid ML scientists and ML engineers.


### Unsure what you will end up with

biased

data poisoning


## When to use active learning?

This post aims to depict active learning in a less romanticized way than
what you usually find in blogposts and scientific publications. Despite the numerous
limitations I listed, I think active learning is a necessity for some projects;
however, it should not be used the way it was originally intended.

In my opinion, **Active Learning be applied when bootstrapping a dataset**
to reduce cost or to have a faster time to model release. These techniques
only **outweigh the drawbacks when building enormous datasets at scale or when
tackling the long tail distribution of edge cases.** Ignoring the tail of edge cases
can be detrimental for some companies.

<h4 class="no_toc">The data infrastructure comes first</h4>

As briefly discussed before, the infrastructure to support fast iterations and curation of
the data is a key component for a successful data strategy. This means the company
needs to invest a lot in its tooling and automation for the data pipeline.
These investments in themselves should be beneficial by speeding up delivery, reducing
human operations, etc.

**Active Learning comes into play only when the data infrastructure
is matured and mastered**. At this scale, active learning becomes the solution to keep
the dataset growth under control and manageable infrastructure costs.

<h4 class="no_toc">The long tail distribution of edge cases</h4>

At first, the model is trying to build a relevant and effective representation
of the domain space. At this stage, feeding more and more data to a machine learning
model yields great improvements.

But once the model already gained great generalization capabilities, it needs to handle edge cases.
By definition, edge cases are rare and difficult. This means **the improvements get smaller
and smaller as the dataset grows**. This phenomenon is known as diminishing returns on data.

There are two ways to tackle this problem. The first one is to make the dataset
several orders of magnitude larger. The bad news is that
[Google](https://ai.googleblog.com/2017/07/revisiting-unreasonable-effectiveness.html),
[Facebook](https://engineering.fb.com/2018/05/02/ml-applications/advancing-state-of-the-art-image-recognition-with-deep-learning-on-hashtags/), and
[Alibaba](https://arxiv.org/abs/2104.10972) have found a
**logarithmic correlation between performances and  the number of annotated samples**.
If you have deep pockets, racks full of GPUs, and a lot of time, then this is an easy solution
to boost performances.

The fact that there is a logarithmic relationship between the number of samples and
the performances is related to the distribution of edge cases. The real world is messy
and therefore all machine learning problems have a long tail distribution of data
(a VERY long tail for most tasks) as illustrated below. As the number
of sample decreases so does the performances.
A16z has a fascinating blog on this topic:
[Taming the Tail: Adventures in Improving AI Economics](https://a16z.com/2020/08/12/taming-the-tail-adventures-in-improving-ai-economics/).
The takeaway is to tackle the long tail
distribution, one should optimize the model, narrow down the problem, and
get more data from customers.


<figure>
  <img src="/assets/img/long-tail-distribution.svg" alt="Long tail distribution illustration">
</figure>

Another solution is to **be more selective when building the training set**.
The goal is to change the prevalence of observations through a careful selection.
Rather than including all possible data points and hoping for the best, it's time
to optimize for the quality of samples. This is where active learning thrives!
It helps identify underrepresented cases where the model struggles,
and it discards samples related to already learned concepts.


## Conclusion

Active Learning is often presented as a way to help bootstrap a dataset.
In my opinion, active learning and other sophisticated data curation
should only be used in a later stage when the data pipeline is already quite advanced.
**Active learning is as much of an engineering solution for dataset scaling as it is
a machine learning technique to chase the last few points of accuracy**.

These techniques, should identify gaps in the datasets, balance data distribution, and
most importantly close the feedback loop. With active learning, it becomes easier to
identify the failure modes of the model. This assessments provide insights
for the exploration and curation of the data, and finally, become ad hoc rules for adding
specific samples to the training set. Being pro-active on the discovery of infrequent
samples is utterly important for most real world applications like self-driving cars
and healthcare.


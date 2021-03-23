---
layout: post
title: "Relationship Between Machine Learning Methods"
date: 2021-03-21 11:36:00 +0100
---

Back when I was doing my master in Data Science and Cognitive Systems at KTH I had an 
interesting conversion about the number of algorithms a data scientist should know to do his work.
My friend was baffled by the number of techniques that we had to learn for the exams.
As a rather lazy student, I was a bit perplex too. The truth is that many methods and tricks can be used
within widely different models.

In that regard, data science is a bit like cooking. Data scientists know some algorithms like a
chef know some recipes. Both of them can follow the step-by-step instructions to complete the job.
However, 3 Michellin-starred chefs not only know many recipes, but they understand how to combine ingredients
and techniques to create new flavours and textures. In the same way, good data scientists can combine
methods and techniques to create different models and boost performances.

With this in mind, let's look at how some models relate.
For example, all models can be used for both classification and regression with a slight modification
to the computed output or the final voting. Another interesting example is to think of
gaussian processes as linear regression with the kernel method under a Bayesian framework.

<figure>
  <img src="/assets/img/model-relationship.svg" alt="Statistical model relationship" style="max-width:680px">
  <figcaption>Fig. 1 — Statistical Model Relationship.</figcaption>
</figure>

The figure above is a small representation of the relationship between machine learning models.
It is far from being exhaustive as it would rapidly get very messy. For instance, the kernel method is
a core element of *support vector machines (SVM)* but can also be used with *k-nearest neighbours (k-NN)*.
Both of them can be used for classification and regression tasks.

Throwing data to an algorithm is not enough. Understanding how statistical methods relate is key
to designing efficient machine learning models and data pipelines.

PS: Here is a small exercise for the reader: among the models in Fig.1 above, which are parametric and non-parametric?

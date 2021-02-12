---
layout: post
title: "K-Nearest Neighbors (KNN) - Visualizing the Variance"
date: 2021-02-12 19:25:00 +0100
---

In chapter 2 of [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/),
the authors compare least square regression and nearest neighbors in terms of bias and variance.
I thought it would be interesting to have an interactive visualization to understand
why k-NN is said to have low bias and high variance.

Section 2.5 starts like this:

> The linear decision boundary from least squares is very smooth, and apparently stable to fit. It does appear to rely heavily on the assumption that a linear decision boundary is appropriate. [...] It has low variance and bigh bias.
<br> On the other hand, the *k-nearest-neighbor* procedures do not appear to rely on any stringent assumptions about the underlying data, and can adapt to any situation. However, any particular subregion of the decision boundary depends on a handful of input points and their particular positions, and is thus wiggly and unstable--high variance and low bias.

For this example, we use k-NN in the context of binary classification where the assignment
is determined by the majority in the voting process. The distance used is the Euclidean distance.

The visualization is interactive. You can:
- adjust the number of neighbors (k) to incorporate in the voting process.
- flip the class of a data point by clicking on it,
- move your mouse to see the nearest neighbors. On mobile, you can also tap anywhere,
- grad a data point and place it wherever you want.
  It's possible that on mobile the drag interfere with the scroll. The trick is to tap, stay in place
  a fraction of a second and only then move the finger.
  If you are on mobile, I recommend you increase the radius parameter,

Have some fun!

<iframe width="100%" height="928" frameborder="0"
  src="https://observablehq.com/embed/@antoinebrl/knn-visualizing-the-variance?cells=viewof+k%2Cviewof+n%2Cviewof+radius%2Cviewof+boundary_cell_size%2Cchart%2Cstyle"></iframe>

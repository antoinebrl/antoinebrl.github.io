---
layout: post
title: Landing on Mars with Reinforcement Learning
date: 2022-02-06 15:14:00 +0100
toc: true
description: "Once in a while, I get dragged back to CodinGame by a friend
who is either learning to code or tackling complex problems.
However, there is one type of challenge I tend to put asides:
optimal control systems. 
This kind of problem often requires hand-crafting a good cost function
and modeling the transition dynamics.
But what if we could solve the challenge without coding
a control policy? This is the story about how I landed a rover on 
Mars with reinforcement learning."
---



## CodinGame Mars Lander

For [this game](https://www.codingame.com/multiplayer/optimization/mars-lander),
the goal is to land the spacecraft while using as few propellants as possible.
The mission is only successful if the rover reaches the flat ground,
at a low speed, and without any tilt.

For each environment, you are given the Mars surface as pairs of coordinates and
the current state of the lander. The state includes position, speed,
angle, current engine thrust, and quantity of remaining fuel. At each iteration,
the program has less than 100ms to output the desired rotation and thrust power.

One test case of the game looks like this: 

<video width="80%" controls style="margin:0 auto 2em auto;display:block">
  <source src="/assets/videos/marslander.mov" type="video/mp4">
Your browser does not support the video tag.
</video> 


## Coding the Game

The CodinGame platform is not designed to run millions of simulations,
gather feedback and improve on them.  There is only one way to
circumvent this limitation: reimplementing the Mars lander game.

### The Interface

As I wanted to do some reinforcement learning, I decided to follow the interface
of the [Environment](https://github.com/openai/gym/blob/c6b6754b128c095df49c74785277d8d5e9f81755/gym/core.py#L17)
class of the [Gym package](https://gym.openai.com/)
developed by [Open AI](https://openai.com/).
Therefore, many RL algorithms can be used out of the box (or so I thought ...).

All Gym Environments follow the same interface with two fields and three methods:
 - `action_space`
 - `observation_space`
 - `reset()`: called to generate a new game
 - `step(action)`: return the new observations of the environment given a specific action
 - `render()`: visualize the agent in his environment

At first, I thought I could go without implementing `render` but I was wrong.
Like in any other machine learning task, visualization is of the utmost
importance for debugging.

### Action Space

The algorithm controls the thrust and the orientation of the spacecraft.
The thrust takes 5 levels between 0 and 4, and the angle is expressed in
degrees between -90 and 90.

Rather than working in absolutes values, I decided that the action space
would be a relative change in thrust and angle. The engine only supports
a +/-1 change in thrust and the orientation cannot change more than
15 degrees in absolute value.

Thrust and angle could be represented with categorical variables, with 3
and 31 mutually exclusive values respectively. Another way, which I decided
to use, is to represent the action space as two continuous variables.
For stability during the training, these values were normalized between -1 and 1.

### Observation Space and Policy Network

The definition of the observation space is a bit more challenging than for
the action space. First, the agent expects a fixed-length input but the
ground is provided as a broken line with up to 30 points. To fulfill this
requirement, I iteratively break down the longest segment into two by
adding an intermediate point until I reach a total of 30 2D points.

To this 60 elements, I concatenated the 7 values which fully characterize the
dynamic of the rover: x and y position, horizontal and vertical speed, angle,
current thrust, and remaining fuel quantity. All these values were normalized
between 0 and 1, or -1 and 1 if negative values are allowed.

In the early experiments, It was clear the fully connected policy was
struggling to identify the landing pad. The rover exhibited completely
different behavior when the landing area was translated horizontally.
The reason is, once breaking down the surface into 30 points, a slight offset
creates a completely different representation of the ground.

Changing the policy to a convolutional neural network (CNN) could help
identify the landing area. By design, unlike MLP, CNNs are translation
equivariant. In addition, CNNs could also have eliminated the problem of
fixed length input we addressed above as their number of
parameters is independent of the input size.

After a few trials, it was clear that this approach would require a lot
more effort to be successful. When using a CNN to extract an abstract
representation of the ground, at some point these features need to be merged
with the rover state. 
When should they be merged? What should be the capacity of the CNN compared
to the MLP? How to initialize both networks? Would they work with
the same learning rate? I ran a few experiments but none was really conclusive.

In the end, to avoid going mad, I decided to use a policy based on MLP
but to help the agent by proving a bit more information.
The trick was to add the x and y coordinates of the two extremities
of the flat section. This extra information can easily be computed by hand
so why not feeding it to the model.

### Simulation

When implementing the `reset()` function, I wanted to get as many possible
environments as possible. I thought generating a random ground surface
and a random rover state would do the job.

It turns out not all these environments could be solved. For example,
the rover might exit the frame before compensating for the initial speed;
or the solution might consume an extremely high volume of propellant.
Finding a valid initial state might be as hard as solving the problem itself.

It was clear that these unsolvable cases were armful during training.
This comes as no surprise, the same rule is true for other machine learning
tasks and algorithms. In the end, I decided to start from the 5 test cases
provided by CodinGame and to apply some random augmentations.

## Reinforcement Learning

### Policy Gradient

For the learning phase, I used [Proximal Policy Optimization, a.k.a PPO](https://arxiv.org/abs/1707.06347).
Despite being less popular than [DQN](https://arxiv.org/abs/1312.5602), PPO performs
better with fewer parameters tuning. In its [revisited version](https://arxiv.org/abs/2009.10897),
PPO introduces a simple twist to the vanilla policy gradient method by
clipping the update magnitude. These smaller optimization steps ensure
a more stable learning.

If you want to learn more about PPO and its siblings, I recommend this article titled
[Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
by Lilian Weng, Applied AI Research Lead at OpenAI.

### Action Noise

In the first experiments, the rover was very shaky as its tilt oscillates
around the optimal value. To eliminate this jerky motion pattern, I used
gSDE which stands for [generalized State-Dependent Exploration](https://arxiv.org/abs/2005.05719).

In reinforcement learning, it's important to balance exploitation with exploration.
Without exploration of the action space, there is no way for the agent to find
potential improvements. The exploration is often achieved by adding random
Gaussian noise to the action distribution.

gSDE authors propose to have a state-dependent noise. Thus, during one episode,
rather than oscillating around a mean value, the action stays the same for a given state. 
This exploration technique leads to smoother trajectories.

### Hyper parameters

In computer vision or natural language processing, it is a good practice
to start the training with some pretrained weights. This speeds up training
and tends to improve performances. However, for this project, there is
no available pretrained model.

The element which is a bit more annoying is the absence of good hyper-parameters.
At first, I went for the default values but the training suffered from
catastrophic forgetting. As you can see on the graph below, the mean reward
drops sharply and struggles to recover.

<figure>
  <img src="/assets/img/rl-mars-lander-unstable-learning.png" alt="Catastrophic forgetting" style="width:100%;max-width:680px">
  <figcaption>Mean Episode Reward - Catastrophic forgetting.</figcaption>
</figure>

Rather than starting an extensive hyper-parameter search, I took inspiration
from the [RL baselines3 Zoo configuration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml).
After a few tweaks I had a great set of values but no doubt
the policy can be further improved with hyper-parameter optimization.


## Export

Training a good policy is only half the work. The final objective is
to submit a solution on CodinGame. Two hurdles made it non-trivial:
Pytorch is not supported and the submission must be shorter than 100k characters.

### Pytorch Modules without Pytorch

TBC

### Encoding

TBC

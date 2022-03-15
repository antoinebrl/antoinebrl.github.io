---
layout: post
title: Landing on Mars with Reinforcement Learning
date: 2022-02-06 15:14:00 +0100
toc: true
description: "Once in a while, I get dragged back to CodinGame by a friend
who is either learning to code or tackling complex problems.
However, there is one type of challenge I tend to put asides:
optimal control systems. This kind of problems often requires to hand-craft
a good cost function and to model the transition dynamics.
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
angle, current engine thrust, and volume of remaining fuel. At each iteration,
the program has less than 100ms to output the desired rotation and thrust power.

One test case of the game looks like this: 

<figure>
    <video width="80%" controls style="margin:0 auto 2em auto;display:block">
      <source src="/assets/videos/marslander.mov" type="video/mp4">
    Your browser does not support the video tag.
    </video>
    <figcaption>Vid 1 — CodinGame Mars Lander</figcaption>
</figure>


## Coding the Game

The CodinGame platform is not designed to run millions of simulations,
gather feedback and improve on them.  There is only one way to
circumvent this limitation: reimplementing the game.

### The Interface

As I wanted to do some reinforcement learning, I decided to follow the interface
of the [Environment](https://github.com/openai/gym/blob/c6b6754b128c095df49c74785277d8d5e9f81755/gym/core.py#L17)
class of the [Gym package](https://gym.openai.com/). Gym is a collection of
test environments to benchmark reinforcement learning algorithms.
By complying to this interface, I can use many algorithms out of the box (or so I thought ...).

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
current thrust, and remaining propellant. All these values were normalized
between 0 and 1, or -1 and 1 if negative values are allowed.

In the early experiments, It was clear the fully connected policy was
struggling to identify the landing pad. The rover exhibited completely
different behavior when the ground was translated horizontally.
The reason is, once breaking down the surface into 30 points, a slight offset
creates a completely different representation of the ground.

Changing the policy to a convolutional neural network (CNN) could help
identify the landing area. By design, unlike MLP, CNNs are translation
equivariant. In addition, CNNs could also have eliminated the problem of
fixed length input we addressed above. Indeed, their number of
parameters is independent of the input size.

After a few trials, it was clear that this approach would require a lot
more efforts to be successful. When using a CNN to extract an abstract
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
environments as possible. I initially thought that generating a random ground surface
and a random rover state would do the job.

It turns out not all these environments could be solved. For example,
the rover might exit the frame before compensating for the initial speed;
or the solution might consume an extremely high volume of propellant.
Finding a valid initial state might be as hard as solving the problem itself.

It was clear that these unsolvable cases were armful during training.
This comes as no surprise, the same rule is true for any other machine learning
tasks and algorithms. In the end, I decided to start from the 5 test cases
provided by CodinGame and to apply some random augmentations.


## Reinforcement Learning

### Policy Gradient

In reinforcement learning, at each time step, an agent interacts with the environment
via its actions. In return, the agent is granted a reward and is place in a new state.
The main assumption is that the future
state only depends on the current state and the action taken. The objective
is to maximise the total reward accumulated over the whole sequence.

There are two popular classes of optimization algorithms: Q-Learning and Policy Gradient.
The former aims at approximating the best transition function from one step
to another. The latter directly optimize in the action space. Despite being
less popular, Policy Gradient methods has the benefit of supporting continuous
action space and tend to converge faster.

For this task, I used a policy gradient method known as
[Proximal Policy Optimization, a.k.a PPO](https://arxiv.org/abs/1707.06347).
In its [revisited version](https://arxiv.org/abs/2009.10897),
PPO introduces a simple twist to the vanilla policy gradient method by
clipping the update magnitude. By reducing the variance and by
taking smaller optimization steps, the learning becomes more stable with
fewer parameters tweaks.

If you want to learn more about PPO and its siblings, I recommend this article titled
[Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
by Lilian Weng, Applied AI Research Lead at OpenAI.

### Action Noise

In the first experiments, the rover was very shaky as its tilt oscillates
around the optimal value. To eliminate this jerky motion pattern, I used
gSDE which stands for [generalized State-Dependent Exploration](https://arxiv.org/abs/2005.05719).

In reinforcement learning, it's important to balance exploitation with exploration.
Without exploration of the action space, there is no way for the agent to find
potential improvements. The exploration is often achieved by adding independent
Gaussian noise to the action distribution.

gSDE authors propose to have a state-dependent noise. Thus, during one episode,
rather than oscillating around a mean value, the action stays the same for a given state. 
This exploration technique leads to smoother trajectories.

<figure>
    <video width="80%" controls style="margin:0 auto 2em auto;display:block">
      <source src="/assets/videos/marslander_no-sde.mov" type="video/mp4">
    Your browser does not support the video tag.
    </video>
    <figcaption>Vid. 2 — Jerky trajectory without gSDE</figcaption>
</figure>

### Reward Shaping

The reward is an incentive mechanism that tells the agent how good it performs.
Crafting this function correctly is a big deal given the goal is to maximize
the cumulated rewards.

#### Reward Sparsity

The first difficulty is reward sparsity. Let's say we want to train a model
to solve a Rubick's cube. What would be a good reward function knowing
there is only one good solution among 43,252,003,274,489,856,000 (~43 quintillion)
possible states?
It would take years to solve if we only rely on luck to reach this state.
If you are interested in this problem, have a look at
[Solving the Rubik's Cube Without Human Knowledge](https://arxiv.org/abs/1805.07470).

In our case, the rover has correctly landed if it grounds on a flat surface,
with a tilt angle of exactly 0°, and a vertical and horizontal speed lower
than 40 m/s and 20 m/s respectively. During training, I decided to loosen up
the angle restriction to anything between -15° and 15°. This increases the
chances of reaching a valid terminal state.
At inference, some post-processing code compensates for the rotation
when the rover is just about to land.

#### Reward Scale

If the rover runs out of fuel or if it leaves the frame, the rover 
gets a negative reward of -150 and the episode ends.
A valid terminal state yields a reward equal to the amount of remaining
propellant.

By default, if the rover is still flying, there is a reward of +1.
In general, positive rewards encourage longer episodes as the agent keep
on accumulating. On the contrary, negative rewards urge the agent
to reach a terminal state as soon as possible to avoid penalties.

For this problem, shorter episodes should consume less fuel.
However, using the quantity of remaining fuel as a terminal reward
creates a massive step function which encourage early completion
of the mission. By keeping a small positive reward at each step the
rover quickly learn to hover.

<figure>
    <video width="80%" controls style="margin:0 auto 2em auto;display:block">
      <source src="/assets/videos/marslander_hover.mov" type="video/mp4">
    Your browser does not support the video tag.
    </video>
    <figcaption>Vid. 3 — Learning to hover</figcaption>
</figure>

For the rest, not all mistakes are made equal. I decided to 
give -75 if speed and angle are correct when touching a non-flat ground;
and -50 if the spacecraft crashes on the landing area. Tough,
without more experiments, it's unclear if the distinction of collisions
brings any advantage.

#### Not Taking Shortcuts

Previously, I talked about helping the model to identify the arrival site.
One idea was to change the structure of the policy by replacing the MLP with a CNN.
The solution I used was to add this information in the input vector. But, a third
possibility was to change the reward function to incorporate a notion of
distance to the landing area.

I ran a few experiments where the reward is the negative
euclidean distance between the landing site and the location of the crash.
It turns out, this strategy instructs the agent to move in a straight
line toward the target.

If there is no direct path between the starting position and the landing site,
then the agent would have to go through a long sequence of decreasing rewards
in order to reach the desired destination. Despite being an intuitive solution,
using the euclidean distance as a negative reward is a strong inductive bias
which reduces the exploration capabilities of the agent.

<figure>
    <video width="80%" controls style="margin:0 auto 2em auto;display:block">
      <source src="/assets/videos/marslander_l2dist.mov" type="video/mp4">
    Your browser does not support the video tag.
    </video>
    <figcaption>Vid. 4 — Suboptimal strategy when minimizing distance to landing site</figcaption>
</figure>

### Hyper-Parameters

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
  <figcaption>Fig 1 — Mean Episode Reward - Catastrophic forgetting.</figcaption>
</figure>

Rather than starting an extensive hyper-parameter search, I took inspiration
from the [RL baselines3 Zoo configurations](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml).
After a few tweaks I had a great set of values but no doubt
the policy can be further improved with hyper-parameter optimization.


## Exporting the Policy Network

Training a good policy is only half the work. The final objective is
to submit a solution on CodinGame. Two hurdles made it non-trivial:
Pytorch is not supported and the submission must be shorter than 100k characters.

### Pytorch Modules without Pytorch

Since v1.7.0, Pytorch has the `fx` subpackage which contains three components:
a symbolic tracer, an intermediate representation, and Python code generation.
By doing a symbolic tracing, we obtain a graph which can be transformed.
Finally, the last bit generates a valid code with the semantic of the graph.

Unfortunately, the code generator only covers the `forward()` method of
the Module. For the `__init__()`, I wrote the code generation to traverse through
all modules and print all parameters weights. Finally, as Pytorch is not available
in the environment, I had to implement three Pytorch modules in pure numpy:
`Sequential`, `Linear` and `ReLU`.

The exported module is self-contained and combines both parameter weights
and computation graph. The result looks something like this:

```python
import numpy as np
from numpy import float32

class MarsLanderPolicy:
    def __init__(self):
        self.policy_net = Sequential(
            Linear(
                weight=np.array([[-6.22188151e-02, -9.03800875e-02,  ...], ...], dtype=float32),
                bias=np.array([-0.03082988,  0.05894076, ...], dtype=float32),
            ),
            ReLU(),
            Linear(
                weight=np.array([[ 6.34499416e-02, -1.32812252e-02,,  ...], ...], dtype=float32),
                bias=np.array([-0.0534077 , -0.02541942, ...], dtype=float32),
            ),
            ReLU(),
        )
        self.action_net = Linear(
            weight=np.array([[-0.07862598, -0.01890517, ...], ...], dtype=float32),
            bias=np.array([0.08827707, 0.10649449], dtype=float32)),
        )

    def forward(self, observation):
        policy_net_0 = getattr(self.policy_net, "0")(observation);  observation = None
        policy_net_1 = getattr(self.policy_net, "1")(policy_net_0);  policy_net_0 = None
        policy_net_2 = getattr(self.policy_net, "2")(policy_net_1);  policy_net_1 = None
        policy_net_3 = getattr(self.policy_net, "3")(policy_net_2);  policy_net_2 = None
        action_net = self.action_net(policy_net_3);  policy_net_3 = None
        return action_net
```


### Encoding

As good as it looks, the exported solution is way too long: it's 440k characters.
We could train a shallower network but let see if we can shrink the generated code by 78%.

The model has 71 input values, two hidden layers with 128 activations and 2 output nodes.
This represents 25,858 free parameters. Each one is a 32 bits float which takes
on average 16 chars in plain text. Even when truncating to float16,
the exported module is only ~100k chars shorter.

Clearly, printing all the decimals of floating numbers is extensive.
A different representation must be used! The shortest solution is obtained by taking the
[base64](https://en.wikipedia.org/wiki/Base64) encoding of the buffers filled with half float.
With this elegant solution, I can finally send a solution as the code is only 72k chars long.


## Conclusion

With a cumulated 2257 liters of fuel left, this solution puts me in 225th place over 4,986 contestants.
The policy only takes between 8ms and 10ms. 

This project was quite fun and a great opportunity for some hands-on practice of RL.
For me, the main take-away is that RL uses the same nuts and bolts than other ML projects.
Simplify the problem, always use visualization, provide good input and don't expect default
hyper-parameters to work on your task.

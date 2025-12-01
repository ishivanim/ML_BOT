====================
DEEP LEARNING
====================

Overview of Deep Learning
---------------------------
Deep Learning is a subset of machine learning that uses artificial neural networks (ANNs) with multiple layers to model complex patterns in data. Deep neural networks can automatically learn hierarchical feature representations.

Key components:
- Input Layer: Receives raw data
- Hidden Layers: Perform transformations and learn features
- Output Layer: Produces predictions
- Activation Functions: Introduce non-linearity (ReLU, Sigmoid, Tanh)
- Loss Function: Measures prediction error (MSE, Cross-Entropy)
- Optimizer: Updates weights (SGD, Adam, RMSProp)

Artificial Neural Networks (ANN)
-----------------------------------
**Introduction**
An Artificial Neural Network (ANN) is a computational model inspired by the human brain.
It consists of many simple processing units called neurons that work together to learn relationships between inputs and outputs.

ANNs are part of the broader field of Deep Learning, a subset of Machine Learning.

WHY ANN?

    - Traditional ML models struggle when:
    - Data is high-dimensional
    - Patterns are complex or non-linear
    - Feature engineering is difficult

ANNs automatically learn hierarchical features and approximate any continuous function (Universal Approximation Theorem).

BASIC BUILDING BLOCK: A NEURON

A single neuron computes:
$$z = \sum_{i=1}^{n} w_i x_i + b,\qquad a = \phi(z)$$

Where:
xᵢ = input features
wᵢ = weights
b = bias
z = linear combination
φ(.) = activation function
a = output of neuron

**Architecture**
ANNs consist of 3 types of layers:

(a) Input Layer
    - Takes raw data (features). No computation.

(b) Hidden Layers
    - Multiple layers containing neurons.
    - They learn intermediate representations of the data.

(c) Output Layer
    - Produces the final result:
    - Regression → single continuous value
    - Binary classification → sigmoid
    - Multiclass classification → softmax

**HOW ANN LEARNS — TRAINING PIPELINE**
(1) Forward Propagation
    - Input → hidden layers → output.
    - Each layer transforms data using weights + activations.

(2) Loss Calculation
    - Compares predicted output and true output.
    - Common loss functions:
    - MSE (Mean Squared Error) – regression
    - Binary Cross Entropy – binary classification
    - Categorical Cross Entropy – multiclass

(3) Backpropagation

    Backprop uses the chain rule of calculus to compute how much each weight contributed to the error.
    It gives gradients (rate of change of loss with respect to weights).

(4) Optimization (Gradient Descent)

    Weights are updated to reduce the loss:
$$w = w - \eta \frac{\partial L}{\partial w}$$

Where η = learning rate.

Popular optimizers:
    - SGD
    - Momentum
    - RMSProp
    - Adam (most widely used)

(5) Iteration (Epochs + Batches)

    Batch: subset of training samples used for one update
    Epoch: one full pass through the dataset
Training = repeating forward + backward + update many times.

**HYPERPARAMETERS**

ANNs have many settings you adjust before training:

    - Number of layers
    - Neurons per layer
    - Activation functions
    - Learning rate
    - Batch size
    - Number of epochs
    - Dropout rate
    - Weight initialization method

**Mathematical Formulation**
    a^{(l)} = f(W^{(l)} a^{(l-1)} + b^{(l)})
where a^(l) is activation, W^(l) weights, b^(l) bias, f() activation

**Backpropagation**
1. Forward pass to compute predictions
2. Compute loss
3. Compute gradients using chain rule
4. Update weights

**Python Implementation (Keras)**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

**Backpropagation**
    Backpropagation, short for Backward Propagation of Errors, is a key algorithm used to train neural networks by minimizing the difference between predicted and actual outputs. It works by propagating errors backward through the network, using the chain rule of calculus to compute gradients and then iteratively updating the weights and biases. Combined with optimization techniques like gradient descent, backpropagation enables the model to reduce loss across epochs and effectively learn complex patterns from data.
    ![Backpropagation](back.webp)

    Back Propagation plays a critical role in how neural networks improve over time. Here's why:

        - Efficient Weight Update: It computes the gradient of the loss function with respect to each weight using the chain rule making it possible to update weights efficiently.
        - Scalability: The Back Propagation algorithm scales well to networks with multiple layers and complex architectures making deep learning feasible.
        - Automated Learning: With Back Propagation the learning process becomes automated and the model can adjust itself to optimize its performance.

    Working of Back Propagation Algorithm

    The Back Propagation algorithm involves two main steps: the Forward Pass and the Backward Pass.

    1. Forward Pass Work

        In forward pass the input data is fed into the input layer. These inputs combined with their respective weights are passed to hidden layers. For example in a network with two hidden layers (h1 and h2) the output from h1 serves as the input to h2. Before applying an activation function, a bias is added to the weighted inputs.

        Each hidden layer computes the weighted sum (`a`) of the inputs then applies an activation function like ReLU (Rectified Linear Unit) to obtain the output (`o`). The output is passed to the next layer where an activation function such as softmax converts the weighted outputs into probabilities for classification.

    2. Backward Pass

        In the backward pass the error (the difference between the predicted and actual output) is propagated back through the network to adjust the weights and biases. One common method for error calculation is the Mean Squared Error (MSE) given by:

            MSE=(Predicted Output−Actual Output)^2

        Once the error is calculated the network adjusts weights using gradients which are computed with the chain rule. These gradients indicate how much each weight and bias should be adjusted to minimize the error in the next iteration. The backward pass continues layer by layer ensuring that the network learns and improves its performance. The activation function through its derivative plays a crucial role in computing these gradients during Back Propagation.

    **Mathematical Formulation**
    We consider a general L-layer feedforward neural network:
1. Network Definitions
    For layer l:
Input activations: $$a^{[l-1]} \in \mathbb{R}^{n_{l-1}}$$
Weights : $$W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$$
Biases : $$b^{[l]} \in \mathbb{R}^{n_l}$$
Linear transformation : $$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
Activation : $$a^{[l]} = f^{[l]}\!\left( z^{[l]} \right)$$
Final output : $$\hat{y} = a^{[L]}$$
Loss : $$L(\hat{y},\, y)$$

2. Goal

Compute gradients:
$$\frac{\partial L}{\partial \mathbf{W}^{[l]}}$$,
$$\frac{\partial L}{\partial \mathbf{b}^{[l]}}$$

for all layers l = 1,2,....,L.

Backprop works by computing error signals:
$$\delta^{[l]} \equiv \frac{\partial L}{\partial \mathbf{z}^{[l]}}
$$
and propagating them backward.

3. STEP 1 — Error at Output Layer

Assume loss L depends on output $$\hat{y} = a^{[L]}$$.

Using chain rule:
$$\delta^{[L]} 
= \frac{\partial L}{\partial \mathbf{a}^{[L]}}
\odot f'(\mathbf{z}^{[L]})
$$

where $\odot$ = element-wise multiplication.

4. STEP 2 — Error Propagation to Hidden Layers
For layer l<L:
$$\delta^{[l]} 
= \left( \mathbf{W}^{[l+1]^\top} \delta^{[l+1]} \right)
\odot f'(\mathbf{z}^{[l]})
$$

This is the key backprop formula.

Interpretation:
Error at layer l = (next layer’s error projected back) × derivative of activation.

5. STEP 3 — Gradients w.r.t. Weights and Biases

Weights:
$$\frac{\partial L}{\partial \mathbf{W}^{[l]}}
= \delta^{[l]} \, \mathbf{a}^{[l-1]^\top}
$$

Biases:
$$\frac{\partial L}{\partial \mathbf{b}^{[l]}}
= \delta^{[l]}$$

(For batch training: sum over samples, or average.)

7. DERIVATION of Gradient

We show how the formula for $\delta^{[l]}$ is derived.
$$
\[
\begin{aligned}
\delta^{[l]} 
&= \frac{\partial L}{\partial \mathbf{z}^{[l]}} \\[6pt]
&= \frac{\partial L}{\partial \mathbf{a}^{[l]}}
   \odot 
   \frac{\partial \mathbf{a}^{[l]}}{\partial \mathbf{z}^{[l]}} \\[6pt]
&= \frac{\partial L}{\partial \mathbf{a}^{[l]}}
   \odot f'(\mathbf{z}^{[l]}) \\[6pt]
\frac{\partial L}{\partial \mathbf{a}^{[l]}}
&= \mathbf{W}^{[l+1]^\top}\delta^{[l+1]} \\[6pt]
\therefore\quad
\delta^{[l]} 
&= (\mathbf{W}^{[l+1]^\top}\delta^{[l+1]})
   \odot f'(\mathbf{z}^{[l]})
\end{aligned}
\]
$$

8. SOFTMAX + CROSS-ENTROPY (SPECIAL CASE)

This is extremely important.

Output layer activation:
$a_i^{[L]} = \frac{e^{z_i^{[L]}}}{\sum_j e^{z_j^{[L]}}}$

Loss : $L = -\sum_i y_i \log a_i^{[L]}$

MAGIC RESULT (very famous) :
$\delta^{[L]} = a^{[L]} - y$

### DIFFERENT TYPES OF NEURAL NETWORKS:

    1. Convolutional Neural Networks (CNN)
    --------------------------------------
    **Introduction**
    Convolutional Neural Network (CNN) is an advanced version of artificial neural networks (ANNs), primarily designed to extract features from grid-like matrix datasets. This is particularly useful for visual datasets such as images or videos, where data patterns play a crucial role. CNNs are widely used in computer vision applications due to their effectiveness in processing visual data.

CNNs consist of multiple layers like the input layer, Convolutional layer, pooling layer, and fully connected layers. Let's learn more about CNNs in detail. 

    Mathematical Overview of Convolution

    Now let’s talk about a bit of mathematics that is involved in the whole convolution process. 

    - Convolution layers consist of a set of learnable filters (or kernels) having small widths and heights and the same depth as that of input volume (3 if the input layer is image input).
    - For example, if we have to run convolution on an image with dimensions 34x34x3. The possible size of filters can be axax3, where ‘a’ can be anything like 3, 5, or 7 but smaller as compared to the image dimension.
    - During the forward pass, we slide each filter across the whole input volume step by step where each step is called stride (which can have a value of 2, 3, or even 4 for high-dimensional images) and compute the dot product between the kernel weights and patch from input volume.
    - As we slide our filters we’ll get a 2-D output for each filter and we’ll stack them together as a result, we’ll get output volume having a depth equal to the number of filters. The network will learn all the filters.

    Layers Used to Build ConvNets

    A complete Convolution Neural Networks architecture is also known as covnets. A covnets is a sequence of layers, and every layer transforms one volume to another through a differentiable function. 

    Let’s take an example by running a covnets on of image of dimension 32 x 32 x 3. 

    - Input Layers: It’s the layer in which we give input to our model. In CNN, Generally, the input will be an image or a sequence of images. This layer holds the raw input of the image with width 32, height 32, and depth 3.

    - Convolutional: This is the layer, which is used to extract the feature from the input dataset. It applies a set of learnable filters known as the kernels to the input images. The filters/kernels are smaller matrices usually 2x2, 3x3, or 5x5 shape. it slides over the input image data and computes the dot product between kernel weight and the corresponding input image patch. The output of this layer is referred as feature maps. Suppose we use a total of 12 filters for this layer we’ll get an output volume of dimension 32 x 32 x 12.

    - Activation : By adding an activation function to the output of the preceding layer, activation layers add nonlinearity to the network. it will apply an element-wise activation function to the output of the convolution layer. Some common activation functions are RELU: max(0, x),  Tanh, Leaky RELU, etc. The volume remains unchanged hence output volume will have dimensions 32 x 32 x 12.

    - Pooling: This layer is periodically inserted in the covnets and its main function is to reduce the size of volume which makes the computation fast reduces memory and also prevents overfitting. Two common types of pooling layers are max pooling and average pooling. If we use a max pool with 2 x 2 filters and stride 2, the resultant volume will be of dimension 16x16x12. 

    ![convolution](maxpool.png)

    - Flattening: The resulting feature maps are flattened into a one-dimensional vector after the convolution and pooling layers so they can be passed into a completely linked layer for categorization or regression.

    - Fully connected:  It takes the input from the previous layer and computes the final classification or regression task.

    - Output Layer: The output from the fully connected layers is then fed into a logistic function for classification tasks like sigmoid or softmax which converts the output of each class into the probability score of each class.

**Mathematical Formulation**
1. 2D Convolution

Given:
    Input image: $X \in \mathbb{R}^{H \times W}$
    Kernel/filter: $K \in \mathbb{R}^{k_H \times k_W}$
    Output feature map:Y

The 2D convolution without padding and stride is:
$$Y(i,j) =
\sum_{m=0}^{k_H - 1}
\sum_{n=0}^{k_W - 1}
X(i+m, j+n)\, K(m,n)
$$

2. Convolution With Stride

Stride s controls how far the filter moves.
$Y(i,j) = 
\sum_{m=0}^{k_H-1}
\sum_{n=0}^{k_W-1}
X(i \cdot s + m,\; j \cdot s + n)\, K(m,n)
$$

3. Convolution With Padding

Padding p enlarges input by adding zeros.

Output dimensions:
$H_{out} = 
\left\lfloor \frac{H + 2p - k_H}{s} \right\rfloor + 1
$

$W_{out} =
\left\lfloor \frac{W + 2p - k_W}{s} \right\rfloor + 1
$

4. Dilation in Convolution

Dilation spacing factor d enlarges the receptive field without adding parameters.

Effective kernel size: $k_{\text{eff}} = d(k - 1) + 1$

5. Multi-Channel Convolution

For RGB or multi-feature inputs:
    Input image: $X \in \mathbb{R}^{C \times H \times W}$
    Kernel/filter: $K \in \mathbb{R}^{C \times k_H \times k_W}$

$$Y(i,j)
=
\sum_{c=1}^{C}
\sum_{m=0}^{k_H - 1}
\sum_{n=0}^{k_W - 1}
X(c, i+m, j+n)\, K(c,m,n)
$$

6. Number of Parameters in a Convolution Layer

For:
    Input channels: ${C_in}$
    Output channels: ${C_out}$
    Kernel size: ${k_H × k_W}$

$\text{params} =
C_{out}\, (C_{in} \cdot k_H \cdot k_W + 1)
$

7. Receptive Field (RF)

The receptive field indicates how many input pixels influence a single output pixel.

For layer l:
$RF_l = RF_{l-1} + (k_l - 1)\prod_{i=1}^{l-1} s_i$

8. Backpropagation Through Convolution
8.1. Gradient w.r.t Kernel Weights
$\frac{\partial L}{\partial K(m,n)} =
\sum_{i,j}
\frac{\partial L}{\partial Y(i,j)} \;
X(i+m,\; j+n)
$

8.2. Gradient w.r.t Input

This is convolution of the gradient with the flipped kernel:
$\frac{\partial L}{\partial X(i,j)} =
\sum_{m,n}
\frac{\partial L}{\partial Y(i-m,\; j-n)}\,
K(m,n)
$

9. Pooling Operations
9.1. Max Pooling
$Y(i,j)=\max_{m,n \in \text{window}} X(i+m, j+n)$

Backprop for Max Pooling
$\frac{\partial L}{\partial X(i,j)} =
\begin{cases}
\frac{\partial L}{\partial Y(p,q)} & \text{if } X(i,j) = \max \\
0 & \text{ otherwise }
\end{cases}
$

9.2. Average Pooling
$Y(i,j) = 
\frac{1}{k_H k_W}
\sum_{m,n}
X(i+m,\; j+n)
$

10. Batch Normalization in CNNs

Given activation a:

Normalization
$\hat{a} = \frac{a - \mu}{\sqrt{\sigma^2 + \epsilon}}$

Scale and Shift : $y = \gamma \hat{a} + \beta$

11. Softmax Function : $\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$

12. Cross-Entropy Loss

For one-hot target y:
$L = -\sum_i y_i \log(\sigma(z_i))$

13. Fully Connected Layer (After Flattening)
z=Wx+b


**Python Implementation (Keras)**
```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```


    2. Recurrent Neural Networks (RNN)
    -----------------------------------
    **Introduction**
    Recurrent Neural Networks (RNNs) differ from regular neural networks in how they process information. While standard neural networks pass information in one direction i.e. from input to output, RNNs feed information back into the network at each step.

    Lets understand RNN with a example:

    Imagine reading a sentence and you try to predict the next word, you don’t rely only on the current word but also remember the words that came before. RNNs work similarly by “remembering” past information and passing the output from one step as input to the next i.e it considers all the earlier words to choose the most likely next word. This memory of previous steps helps the network understand context and make better predictions. 

    **Key Components of RNNs**

    There are mainly two components of RNNs that we will discuss.
    1. Recurrent Neurons

        The fundamental processing unit in RNN is a Recurrent Unit. They hold a hidden state that maintains information about previous inputs in a sequence. Recurrent units can "remember" information from prior steps by feeding back their hidden state, allowing them to capture dependencies across time.

![Recurrent neurons](neuron.png)

    2. RNN Unfolding

        RNN unfolding or unrolling is the process of expanding the recurrent structure over time steps. During unfolding each step of the sequence is represented as a separate layer in a series illustrating how information flows across each time step.

        This unrolling enables backpropagation through time (BPTT) a learning process where errors are propagated across time steps to adjust the network’s weights enhancing the RNN’s ability to learn dependencies within sequential data.

![Recurrent Unfolding](memory.png)

    **Recurrent Neural Network Architecture**

    RNNs share similarities in input and output structures with other deep learning architectures but differ significantly in how information flows from input to output. Unlike traditional deep neural networks where each dense layer has distinct weight matrices. RNNs use shared weights across time steps, allowing them to remember information over sequences.

    In RNNs the hidden state HiHi​​ is calculated for every input XiXi​​ to retain sequential dependencies. The computations follow these core formulas:

    1. Hidden State Calculation:

        h=σ(U⋅X+W⋅h_(t−1)+B)

    Here:

        h represents the current hidden state.
        U and WW are weight matrices.
        B is the bias.

    2. Output Calculation:

        Y=O(V⋅h+C) 

    The output Y is calculated by applying O an activation function to the weighted hidden state where V and C represent weights and bias.

    3. Overall Function:

        Y=f(X,h,W,U,V,B,C) 

    This function defines the entire RNN operation where the state matrix S holds each element s_i representing the network's state at each time step i.

![RNN](rnn.png)

    *How does RNN work?*

    At each time step RNNs process units with a fixed activation function. These units have an internal hidden state that acts as memory that retains information from previous time steps. This memory allows the network to store past knowledge and adapt based on new inputs.

1. Structure of a Basic RNN

At each time step t:
    Input: ${x_i} \in \mathbb{R}^{d}$
    Hidden state : ${h_t} \in \mathbb{R}^{h}$
    Output : ${y_t} \in \mathbb{R}^{o}$

    The RNN maintains memory via recurrent connection:
$h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$

where:
$W_{xh} \in \mathbb{R}^{h \times d}$
$W_{hh} \in \mathbb{R}^{h \times h}$
Activation f is usually tanh or ReLU.

2. Output Computation
$y_t = W_{hy} h_t + b_y$


3. Sequence Processing

Given a sequence:

x=(x1,x2,...,xT)

The hidden states evolve as:
$h1 = f(W_{xh}x_1 + W_{hh}h_0 +b_{h})
$h2 = f(W_{xh}x_2 + W_{hh}h_1 +b_{h}) ...
$h_{T} = f(W_{xh}x_{T} + W_{hh}h_{T-1} +b_{h})

4. Activation Functions Used
Tanh : $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
Sigmoid : $\sigma(z) = \frac{1}{1 + e^{-z}}$

5. Loss Function for RNN

For sequence output with target $y_t^{*}$:
$L = \sum_{t=1}^{T} \ell(y_t, y_t^{*})$

Most common choice: Cross-Entropy Loss
$\ell(y, y^*) = -\sum_i y_i^* \log(y_i)$

6. Backpropagation Through Time (BPTT)

RNNs unroll over time during backpropagation.

We compute gradients from last step T to first step 1.

6.1 Gradient w.r.t Output Weights

$\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L}{\partial y_t} h_t^{\top}$

6.2 Gradient w.r.t Hidden State
$\delta_t = 
\frac{\partial L}{\partial h_t} +
W_{hh}^{\top} \delta_{t+1} \odot f'(a_t)$

where : 
$a_t = W_{xh}x_t + W_{hh}h_{t-1} +b_{h}

Activation derivatives:

For tanh:$f'(a_t) = 1 - h_t^2$

6.3 Gradients w.r.t RNN Parameters
Gradient w.r.t $W_{xh}$
$\frac{\partial L}{\partial W_{xh}} =
\sum_{t=1}^{T} \delta_t\, x_t^{\top}$

​Gradient w.r.t $W_{hh}$ :
$\frac{\partial L}{\partial W_{hh}} =
\sum_{t=1}^{T} \delta_t\, h_{t-1}^{\top}$

Gradient w.r.t Bias $b_h$:
$\frac{\partial L}{\partial b_h} =
\sum_{t=1}^{T} \delta_t$

7. Vanishing and Exploding Gradients

During BPTT, gradients repeatedly multiply by $W_{hh}$ : 
$\delta_t =
(W_{hh}^{\top})^{T-t}
\frac{\partial L}{\partial h_T}$

If:
∥$W_{hh}$∥<1→ vanishing gradients
∥$W_{hh}$∥>1 → exploding gradients

This explains why vanilla RNNs struggle with long sequences.

9. Parameter Count in RNN

For hidden size h and input size d:

params = h(d+h+1)+o(h+1)

For hidden layer only:

$\text{params}_{RNN} = h(d + h + 1)$


    **Variants**
    There are several variations of RNNs, each designed to address specific challenges or optimize for certain tasks:

    1. Vanilla RNN

        A Vanilla RNN processes a sequence one step at a time.
        At each time step, it takes the current input and the previous hidden state, mixes them, and produces a new hidden state. This hidden state works like a memory.

        Key idea:
        RNNs capture short-term dependencies but struggle with long sequences because gradients vanish as they are backpropagated through many time steps.

**Python Implementation**
``` python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_simple_rnn(input_dim, hidden_units, output_dim):
    model = models.Sequential([
        layers.SimpleRNN(hidden_units, return_sequences=True, input_shape=(None, input_dim)),
        layers.Dense(output_dim)
    ])
    return model

# Example usage:
# model = build_simple_rnn(input_dim=10, hidden_units=32, output_dim=5)
# model.compile(optimizer='adam', loss='mse')

```

    2. LSTM (Long Short-Term Memory): 
        Long Short-Term Memory Networks (LSTMs) introduce a memory mechanism to overcome the vanishing gradient problem. Each LSTM cell has three gates:

        Input Gate: Controls how much new information should be added to the cell state.
        Forget Gate: Decides what past information should be discarded.
        Output Gate: Regulates what information should be output at the current step. This selective memory enables LSTMs to handle long-term dependencies, making them ideal for tasks where earlier context is critical.

        They maintain a separate cell state that carries long-term information.

        Key benefit: keeps memory unchanged over many time steps → captures long-term dependencies.

**Python Implementation**
```python
def build_lstm(input_dim, hidden_units, output_dim):
    model = models.Sequential([
        layers.LSTM(hidden_units, return_sequences=True, input_shape=(None, input_dim)),
        layers.Dense(output_dim)
    ])
    return model

# model = build_lstm(10, 64, 5)

```
    3. GRU (Gated Recurrent Unit): 

        Gated Recurrent Units (GRUs) simplify LSTMs by combining the input and forget gates into a single update gate and streamlining the output mechanism. This design is computationally efficient, often performing similarly to LSTMs and is useful in tasks where simplicity and faster training are beneficial.
        A GRU simplifies LSTM by combining gates:

            - Update gate: how much of the past to keep

            - Reset gate: how much of the past to forget

        No separate cell state — the hidden state alone stores memory.

        Key benefit:
        Faster and lighter than LSTM, but often performs just as well.

**Python Implementation**
```python
def build_gru(input_dim, hidden_units, output_dim):
    model = models.Sequential([
        layers.GRU(hidden_units, return_sequences=True, input_shape=(None, input_dim)),
        layers.Dense(output_dim)
    ])
    return model

# model = build_gru(10, 64, 5)

```

    4. Bi-directional RNN:
        Bidirectional RNNs process inputs in both forward and backward directions, capturing both past and future context for each time step. This architecture is ideal for tasks where the entire sequence is available, such as named entity recognition and question answering.
**Python Implementation**
```python
def build_birnn(input_dim, hidden_units, output_dim):
    model = models.Sequential([
        layers.Bidirectional(layers.SimpleRNN(hidden_units, return_sequences=True),
                             input_shape=(None, input_dim)),
        layers.Dense(output_dim)
    ])
    return model

# model = build_birnn(10, 32, 5)

```

    5. Stacked / Deep RNN (Multi-layer RNN)

        A stacked RNN has multiple recurrent layers placed on top of each other.
        The output of layer 1 becomes the input of layer 2, and so on.

        Benefit:
        Learns more complex hierarchical patterns, similar to deep feedforward networks.
**Python Implementation**
```python
def build_stacked_lstm(input_dim, hidden_units, num_layers, output_dim):
    model = models.Sequential()
    
    # First layer with input shape
    model.add(layers.LSTM(hidden_units, return_sequences=True, input_shape=(None, input_dim)))
    
    # Middle layers
    for _ in range(num_layers - 2):
        model.add(layers.LSTM(hidden_units, return_sequences=True))
    
    # Final LSTM layer
    model.add(layers.LSTM(hidden_units, return_sequences=True))
    
    # Output
    model.add(layers.Dense(output_dim))
    
    return model

# model = build_stacked_lstm(10, 64, 3, 5)

```

    3. Autoencoders
    ----------------
    **Introduction**
    Autoencoders are a special type of neural networks that learn to compress data into a compact form and then reconstruct it to closely match the original input. They consist of an:

    Encoder that captures important features by reducing dimensionality.
    Decoder that rebuilds the data from this compressed representation. 

    The model trains by minimizing reconstruction error using loss functions like Mean Squared Error or Binary Cross-Entropy. These are applied in tasks such as noise removal, error detection and feature extraction where capturing efficient data representations is important.

    **Architecture**
    An autoencoder’s architecture consists of three main components that work together to compress and then reconstruct data which are as follows:
![Autoencoder Architecture](encoder.webp)

    1. Encoder: 
        It compress the input data into a smaller, more manageable form by reducing its dimensionality while preserving important information. It has three layers which are:

        - Input Layer: This is where the original data enters the network. It can be images, text features or any other structured data.

        - Hidden Layers: These layers perform a series of transformations on the input data. Each hidden layer applies weights and activation functions to capture important patterns, progressively reducing the data's size and complexity.

        - Output(Latent Space): The encoder outputs a compressed vector known as the latent representation or encoding. This vector captures the important features of the input data in a condensed form helps in filtering out noise and redundancies.

    2. Bottleneck (latent Space):
        It is the smallest layer of the network which represents the most compressed version of the input data. It serves as the information bottleneck which force the network to prioritize the most significant features. This compact representation helps the model learn the underlying structure and key patterns of the input helps in enabling better generalization and efficient data encoding.

    - Decoder: 
        It is responsible for taking the compressed representation from the latent space and reconstructing it back into the original data form.

        - Hidden Layers: These layers progressively expand the latent vector back into a higher-dimensional space. Through successive transformations decoder attempts to restore the original data shape and details

        - Output Layer: The final layer produces the reconstructed output which aims to closely resemble the original input. The quality of reconstruction depends on how well the encoder-decoder pair can minimize the difference between the input and output during training.

    *Loss Function in Autoencoder Training*

        During training an autoencoder’s goal is to minimize the reconstruction loss which measures how different the reconstructed output is from the original input. The choice of loss function depends on the type of data being processed:

        - Mean Squared Error (MSE): This is commonly used for continuous data. It measures the average squared differences between the input and the reconstructed data.
        - Binary Cross-Entropy: Used for binary data (0 or 1 values). It calculates the difference in probability between the original and reconstructed output.

    During training the network updates its weights using backpropagation to minimize this reconstruction loss. By doing this it learns to extract and retain the most important features of the input data which are encoded in the latent space.

    *Types of Autoencoders*

    Lets see different types of Autoencoders which are designed for specific tasks with unique features:

    1. Denoising Autoencoder

    Denoising Autoencoder is trained to handle corrupted or noisy inputs, it learns to remove noise and helps in reconstructing clean data. It prevent the network from simply memorizing the input and encourages learning the core features.
**Python Implementation**
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class DenoisingEncoder(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(784,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(latent_dim)
        ])
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(784, activation="sigmoid")
        ])

    def call(self, x):
        # add Gaussian noise
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.2)
        noisy_x = x + noise

        z = self.encoder(noisy_x)
        reconstructed = self.decoder(z)
        return reconstructed
```

    2. Sparse Autoencoder

    Sparse Autoencoder contains more hidden units than input features but only allows a few neurons to be active simultaneously. This sparsity is controlled by zeroing some hidden units, adjusting activation functions or adding a sparsity penalty to the loss function.
**Python Implementation**
```python
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class SparseEncoder(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(784,)),
            layers.Dense(256, activation="relu", 
                         activity_regularizer=regularizers.l1(1e-4)),
            layers.Dense(latent_dim, activation="relu",
                         activity_regularizer=regularizers.l1(1e-4))
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(784, activation="sigmoid")
        ])

    def call(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed
```

    3. Variational Autoencoder

    Variational autoencoder (VAE) makes assumptions about the probability distribution of the data and tries to learn a better approximation of it. It uses stochastic gradient descent to optimize and learn the distribution of latent variables. They used for generating new data such as creating realistic images or text.

    It assumes that the data is generated by a Directed Graphical Model and tries to learn an approximation to qϕ(z∣x)qϕ​(z∣x) to the conditional property qθ(z∣x)qθ​(z∣x) where ϕ ϕ and θ θ are the parameters of the encoder and the decoder respectively.
**Python Implementation**
```python
class VAEEncoder(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.d1 = layers.Dense(128, activation='relu')
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

    def call(self, x):
        h = self.d1(x)
        return self.z_mean(h), self.z_log_var(h)

```

    4. Convolutional Autoencoder

    Convolutional autoencoder uses convolutional neural networks (CNNs) which are designed for processing images. The encoder extracts features using convolutional layers and the decoder reconstructs the image through deconvolution also called as upsampling.
**Python Implementation**
```python
def build_cnn_encoder():
    return tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(128)  # encoded vector
    ])
```

**Python Implementation for general encoder**
```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```
![Autoencoder ](en.png)

    4. Generative Adversarial Networks (GAN)
    ----------------------------------------
    **Introduction**
    Generative Adversarial Networks (GAN) help machines to create new, realistic data by learning from existing examples. It is introduced by Ian Goodfellow and his team in 2014 and they have transformed how computers generate images, videos, music and more. Unlike traditional models that only recognize or classify data, they take a creative way by generating entirely new content that closely resembles real-world data. This ability helped various fields such as art, gaming, healthcare and data science.

    **Architecture of GAN**

    GAN consist of two main models that work together to create realistic synthetic data which are as follows:

    1. Generator Model

    The generator is a deep neural network that takes random noise as input to generate realistic data samples like images or text. It learns the underlying data patterns by adjusting its internal parameters during training through backpropagation. Its objective is to produce samples that the discriminator classifies as real.

    Generator Loss Function: The generator tries to minimize this loss:
$J_G = -\frac{1}{m} \sum_{i=1}^{m} \log D(G(z_i))$

where,
$J_G$ measure how well the generator is fooling the discriminator.
$G(z_i)$ is the generated sample from random noise z_i
$D(G(Z_i))$ is the discriminator’s estimated probability that the generated sample is real.

    The generator aims to maximize D(G(zi))D(G(zi​)) meaning it wants the discriminator to classify its fake data as real (probability close to 1).

    2. Discriminator Model

    The discriminator acts as a binary classifier helps in distinguishing between real and generated data. It learns to improve its classification ability through training, refining its parameters to detect fake samples more accurately. When dealing with image data, the discriminator uses convolutional layers or other relevant architectures which help to extract features and enhance the model’s ability.

    Discriminator Loss Function: The discriminator tries to minimize this loss:
$$J_D = -\frac{1}{m} \sum_{i=1}^{m} \log D(x_i)
      - \frac{1}{m} \sum_{i=1}^{m} \log\!\left( 1 - D(G(z_i)) \right)
$$

where,
$J_D$ measures how well the discriminator classifies real and fake samples.
xi​ is a real data sample.
$G(z_i)$ is a fake sample from the generator.
$D(x_i)$ is the discriminator’s probability that xi​ is real.
$D(G(Z_i))$ is the discriminator’s probability that the fake sample is real.

    The discriminator wants to correctly classify real data as real (maximize logD(xi)) and fake data as fake (maximize log(1−D(G(zi))))

    MinMax Loss

    GANs are trained using a MinMax Loss between the generator and discriminator: 
$$\min_G \max_D \; V(G, D)
= \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)]
+ \mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))]
$$

where,
Gis generator network and is DD is the discriminator network.
$p_{\text{data(x)}}$ = true data distribution.
$P_z(Z)$ = distribution of random noise (usually normal or uniform)
D(x) = discriminator’s estimate of real data

    The generator tries to minimize this loss (to fool the discriminator) and the discriminator tries to maximize it (to detect fakes accurately).
![GAN](gan.webp)

   ** How does a GAN work?**

    GAN train by having two networks the Generator (G) and the Discriminator (D) compete and improve together. Here's the step-by-step process
    1. Generator's First Move

    The generator starts with a random noise vector like random numbers. It uses this noise as a starting point to create a fake data sample such as a generated image. The generator’s internal layers transform this noise into something that looks like real data.
    2. Discriminator's Turn

    The discriminator receives two types of data:

        Real samples from the actual training dataset.
        Fake samples created by the generator.

    D's job is to analyze each input and find whether it's real data or something G cooked up. It outputs a probability score between 0 and 1. A score of 1 shows the data is likely real and 0 suggests it's fake.
    3. Adversarial Learning

        If the discriminator correctly classifies real and fake data it gets better at its job.
        If the generator fools the discriminator by creating realistic fake data, it receives a positive update and the discriminator is penalized for making a wrong decision.

    4. Generator's Improvement

        Each time the discriminator mistakes fake data for real, the generator learns from this success.
        Through many iterations, the generator improves and creates more convincing fake samples.

    5. Discriminator's Adaptation

        The discriminator also learns continuously by updating itself to better spot fake data.
        This constant back-and-forth makes both networks stronger over time.

    6. Training Progression

        As training continues, the generator becomes highly proficient at producing realistic data.
        Eventually the discriminator struggles to distinguish real from fake shows that the GAN has reached a well-trained state.
        At this point, the generator can produce high-quality synthetic data that can be used for different applications.

    **Types of GAN**

    There are several types of GANs each designed for different purposes. Here are some important types:

    1. Vanilla GAN

    Vanilla GAN is the simplest type of GAN. It consists of:

        A generator and a discriminator both are built using multi-layer perceptrons (MLPs).
        The model optimizes its mathematical formulation using stochastic gradient descent (SGD).

    While foundational, Vanilla GAN can face problems like:

        Mode collapse: The generator produces limited types of outputs repeatedly.
        Unstable training: The generator and discriminator may not improve smoothly.

    2. Conditional GAN (CGAN)

    Conditional GAN (CGAN) adds an additional conditional parameter to guide the generation process. Instead of generating data randomly they allow the model to produce specific types of outputs.

    Working of CGANs:

        A conditional variable (y) is fed into both the generator and the discriminator.
        This ensures that the generator creates data corresponding to the given condition (e.g generating images of specific objects).
        The discriminator also receives the labels to help distinguish between real and fake data.

    Example: Instead of generating any random image, CGAN can generate a specific object like a dog or a cat based on the label.

    3. Deep Convolutional GAN (DCGAN)

    Deep Convolutional GAN (DCGAN) are among the most popular types of GANs used for image generation.

    They are important because they:

        Uses Convolutional Neural Networks (CNNs) instead of simple multi-layer perceptrons (MLPs).
        Max pooling layers are replaced with convolutional stride helps in making the model more efficient.
        Fully connected layers are removed, which allows for better spatial understanding of images.

    DCGANs are successful because they generate high-quality, realistic images.

    4. Laplacian Pyramid GAN (LAPGAN)

    Laplacian Pyramid GAN (LAPGAN) is designed to generate ultra-high-quality images by using a multi-resolution approach.

    Working of LAPGAN:

        Uses multiple generator-discriminator pairs at different levels of the Laplacian pyramid.
        Images are first down sampled at each layer of the pyramid and upscaled again using Conditional GAN (CGAN).
        This process allows the image to gradually refine details and helps in reducing noise and improving clarity.

    Due to its ability to generate highly detailed images, LAPGAN is considered a superior approach for photorealistic image generation.

    5. Super Resolution GAN (SRGAN)

    Super-Resolution GAN (SRGAN) is designed to increase the resolution of low-quality images while preserving details.

    Working of SRGAN:

        Uses a deep neural network combined with an adversarial loss function.
        Enhances low-resolution images by adding finer details helps in making them appear sharper and more realistic.
        Helps to reduce common image upscaling errors such as blurriness and pixelation.


**Python Implementation (conceptual)**
```python
# Define generator and discriminator models
# Train discriminator on real & fake samples
# Train generator to fool discriminator
```

### Deep Learning Applications
-----------------------------
1. Computer Vision

Deep learning is the backbone of modern vision systems. Convolutional Neural Networks (CNNs) extract spatial features and recognize patterns at multiple scales.

Applications
    - Image Classification – Identifying objects in images (e.g., ImageNet, medical X-rays).
    - Object Detection – Locating and classifying objects (YOLO, Faster R-CNN).
    - Semantic Segmentation – Pixel-level classification (U-Net, DeepLab).
    - Face Recognition – Security, biometrics, authentication systems.
    - Image Super-Resolution – Enhance image quality using GANs.
    - Image Generation – Deep generative models (GANs, Diffusion Models).

2. Natural Language Processing (NLP)

Deep learning models understand and generate human language using embeddings, transformers, and recurrent architectures.

Applications
    - Machine Translation – Google Translate, multilingual LLMs.
    - Sentiment Analysis – Social media insights, brand monitoring.
    - Chatbots & Virtual Assistants – LLMs, customer support bots.
    - Text Summarization – News, legal documents.
    - Speech Recognition – Siri, Alexa, medical dictation.
    - Text Generation – GPT-style models for creative writing and code.

3. Speech, Audio & Signal Processing

    Deep models extract temporal patterns from audio signals, enabling real-time understanding.

    Applications
        - Speech-to-Text – Voice assistants.
        - Voice Cloning – Personalized synthetic voices.
        - Music Generation – AI-generated songs.
        - Noise Reduction – Denoising autoencoders for audio.
        - Emotion detection from voice.

4. Healthcare & Medicine

    Deep learning helps with diagnosis, medical image interpretation, and personalized treatment.

    Applications
        - Disease Detection from X-rays, MRI, CT scans.
        - Drug Discovery – Predict drug-target interactions.
        - Genomics – DNA sequence modeling using transformers.
        - Patient Monitoring – ICU predictions using RNNs.
        - Cancer Detection – Automated pathology analysis.

5. Autonomous Vehicles

    Deep learning enables perception and decision making in self-driving cars.

    Applications
        - Lane detection, traffic sign detection using CNNs.
        - Object tracking for pedestrians, vehicles.
        - Sensor fusion from cameras, LiDAR, radar.
        - Trajectory prediction with RNNs/LSTMs.
        - Control decision-making using reinforcement learning.

6. Robotics

    DL helps robots perceive, plan, and act.

    Applications
        - Grasping and manipulation with visual feedback.
        - SLAM (Simultaneous Localization and Mapping).
        - Human-robot interaction.
        - Autonomous drone navigation.
        - Motion planning using reinforcement learning.

7. Finance

    Deep learning is widely used for prediction, fraud detection, and automation.

    Applications
        - Stock price prediction (time-series models).
        - Fraud detection in transactions.
        - Algorithmic trading with RL.
        - Credit scoring.
        - Risk modeling.

8. Recommendation Systems

    DL-based recommenders outperform classical collaborative filtering models.

    Applications
        - Movie/music recommendations (Netflix, Spotify).
        - E-commerce preferences (Amazon).
        - Content ranking (YouTube, Instagram).
        - Uses deep embeddings + sequence models to learn user behavior.

9. Generative AI

    Generative DL models create data indistinguishable from real content.

    Applications
        - Image Generation – Diffusion models, GANs.
        - Video Generation – Video diffusion, 3D reconstruction.
        - Deepfakes.
        - Style Transfer – Artistic rendering.
        - Text-to-image/video models like DALL·E, Midjourney.

10. Security & Cybersecurity

    Deep models help detect threats from millions of logs.

    Applications
        - Malware classification.
        - Intrusion detection using anomaly detection.
        - Biometric authentication.
        - Phishing detection using NLP.

11. Manufacturing & Industry 4.0

    DL improves automation, monitoring, and optimization.

    Applications
        - Defect detection in industrial products.
        - Predictive maintenance.
        - Process optimization.
        - Robotic assembly.

12. Agriculture

    Smart farming uses DL for monitoring and prediction.

    Applications
        - Crop disease detection from images.
        - Yield prediction.
        - Soil monitoring with IoT + DL.
        - Autonomous tractors & drones.

13. Energy Sector

    Deep learning helps optimize large scale systems.

    Applications
        - Smart grid optimization.
        - Power demand forecasting.
        - Solar and wind prediction.
        - Fault detection in networks.

14. Education

    AI-enabled learning systems adapt to students’ abilities.

    Applications
        - Personalized learning.
        - Exam question generation.
        - Automated grading.
        - AI tutoring systems.

15. Entertainment & Gaming

    DL powers modern gameplay and content generation.

    Applications
        - NPC behavior using RL.
        - Game content generation.
        - Animation synthesis.
        - Film VFX enhancement.

====================
END OF DEEP LEARNING SECTION
====================


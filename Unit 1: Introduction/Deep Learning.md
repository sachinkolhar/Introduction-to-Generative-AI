# Deep Learning

## Introduction

- Deep Learning is a subfield of Machine Learning that focuses on using artificial neural networks to model and solve complex problems. These models operate on massive datasets and learn by extracting hierarchical features from data.

### Why is it Called "Deep"?

The term "Deep" in Deep Learning refers to the multiple layers in artificial neural networks. These layers include:

- **Input Layers**: Receive raw data.
- **Hidden Layers**: Extract different features from the data.
- **Output Layers**: Provide final predictions or classifications.

Each layer in a deep learning model extracts different levels of features:

- **Top Layers**: Capture simple patterns (e.g., edges in images).
- **Deeper Layers**: Capture more complex patterns (e.g., shapes and objects).

### Key Characteristics of Deep Learning

- **Subset of Machine Learning**: Based on artificial neural networks.
- **Hierarchical Learning**: Learns from raw input to complex features.
- **Multiple Layers**: Consists of input, hidden, and output layers.
- **Large Datasets**: Requires massive datasets for effective training.
- **Feature Extraction**: Automates the process of learning patterns from data.

---

##  Neural Networks and Artificial Neural Networks

Deep Learning relies on **Artificial Neural Networks (ANNs)**, which are inspired by the structure and function of the human brain.

### The Human Brain and Neurons

![[Week_01-02.pptx-1.pdf#page=26&rect=702,161,956,394|Week_01-02.pptx-1, p.26]]
6. **Neurons**:
    
    - The human brain consists of approximately 86 billion neurons.
    - Neurons are the fundamental units of the nervous system.
7. **Connections**:
    
    - Neurons are connected to thousands of other neurons via <font color="#ffff00">axons</font>.
    - Axons transmit electrical signals to other neurons.
8. **Input Reception**:
    
    - <font color="#ffff00">Dendrites</font> receive stimuli from the external environment or sensory organs.
    - Dendrites function as the "input receivers" of the neuron.
9. **Electric Impulses**:
    
    - Inputs create electrical impulses called action potentials.
    - These impulses propagate through the neural network rapidly.
10. **Signal Transmission**:
    
    - A neuron can transmit signals to another neuron for further processing.
    - If the stimulus is weak, the neuron may not transmit the signal forward.

---

#### Summary

 Inspired by the human brain, artificial neural networks function by mimicking how neurons process and transmit information. The hierarchical structure enables deep learning models to automatically learn and improve from vast amounts of data, making them highly effective in various applications such as image recognition, speech processing, and natural language understanding


### Artificial Neural Network (ANN)

![[Week_01-02.pptx-1.pdf#page=27&rect=731,163,956,346|Week_01-02.pptx-1, p.27]]

11. **Nodes**:
    
    - ANNs are composed of artificial neurons (also called nodes or units).
        
    - These neurons are mathematical functions that process inputs and produce outputs.
        
12. **Connections**:
    
    - Neurons are connected by <font color="#ffff00">weights</font>, which determine the strength of the connection between nodes.
        
    - These weights are adjusted during training to optimize performance.
        
13. **Input Reception**:
    
    - The input layer receives data (e.g., images, text, numbers) and passes it to the network.
        
    - Each input is multiplied by a corresponding weight before being processed.
        
14. **Signal Transmission**:
    
    - Inputs are processed through weighted sums and activation functions (e.g., ReLU, Sigmoid).
        
    - The output of one neuron becomes the input for the next layer.
        
15. **Activation Function**:
    
    - Each neuron applies an activation function to decide whether to "fire" (send a signal) or not.
        
    - Examples: ReLU, Sigmoid, Tanh.
        
### **Neural Network Architecture**

- **Layers in a Neural Network**
    
    - **Input Layer**: Contains the input features (e.g., x1,x2,x3x_1, x_2, x_3).
    - **Hidden Layer(s)**: Intermediate layer(s) that process data before passing it to the output layer.
    - **Output Layer**: Produces the final prediction or classification output.
- **Connections Between Layers**
    
    - weights
	    - Each node in a layer is connected to nodes in the next layer via **weights**.
    - **bias**
	    - Each node has an associated **bias** term.

### **Example Problem Steps**

- Given **weights, biases, and input values**, compute the output step-by-step:
    1. Compute <font color="#ffff00">weighted sums</font> for the hidden layer.
    2. Apply the <font color="#ffff00">sigmoid activation</font> function to get hidden layer outputs.
    3. Compute the <font color="#ffff00">weighted sum</font> for the output layer.
    4. Apply the <font color="#ffff00">sigmoid function</font> to get the final output.


## Key Components of ANN Architecture


16. **Input Layer**:
    
    - Receives raw input data (e.g., images, text, numerical values).
    - The number of neurons equals the number of input features.
    
17. **Hidden Layers**:
    
    - Intermediate layers between the input and output layers.
    - Perform feature extraction and transformation.
    - Can have one or multiple hidden layers (deep networks).
        
18. **Output Layer**:
    
    - Produces the final result (e.g., classification, regression).
    - Number of neurons depends on the task:
        - Binary classification: 1 neuron.
        - Multi-class classification: 1 neuron per class.
        - Regression: 1 or more neurons.
        
19. **Weights and Biases**:
    
    - **Weights**: Determine the strength of connections between neurons.
    - **Biases**: Allow the model to <font color="#ffff00">shift the activation function for better fit.</font>
        
20. **Activation Functions**:
    
    - Introduce non-linearity to the model.
    - Common functions: ReLU, Sigmoid, Tanh, Softmax.
        
21. **Loss Function**:
    
    - Measures the difference between predicted and actual outputs.
    - Examples: Mean Squared Error (MSE), Cross-Entropy Loss.
        
22. **Optimization Algorithm**:
    
    - <font color="#ffff00">Adjusts weights and biases </font>to minimize the loss function.
    - Examples: Gradient Descent, Adam, RMSprop.
        

---

### Summary

Deep Learning is a powerful subfield of Machine Learning that utilizes artificial neural networks with multiple layers to extract complex patterns from large datasets. Inspired by the human brain, artificial neural networks function by mimicking how neurons process and transmit information. The hierarchical structure enables deep learning models to automatically learn and improve from vast amounts of data, making them highly effective in various applications such as image recognition, speech processing, and natural language understanding.

## Activation Functions 

Activation functions introduce non-linearity into neural networks, helping them learn complex patterns and relationships.

- Without activation functions, neural networks would be restricted to modeling only linear relationships between inputs and outputs. 
- Activation functions introduce non-linearities, allowing neural networks to learn highly complex mappings between inputs and outputs.

---

### **Types of Activation Functions**


- **Sigmoid Activation Function** (for binary classification)
- **Softmax Activation Function** (for multi-class classification)
- **ReLU (Rectified Linear Unit) Activation Function** (default for most deep learning tasks)

---

#### **Sigmoid Activation Function**

<font color="#ffff00">f(x)=1/1+e^-x     </font> 
- Takes a real-valued input and transforms it into a range between **0 and 1**.
- Has an **S-shaped (sigmoid) curve**:
    - **Large negative values** approach **0**.
    - **Large positive values** approach **1**.
- Commonly used in **binary classification** tasks.

![[Week_01-02.pptx-1.pdf#page=34&rect=191,187,622,450|Week_01-02.pptx-1, p.34]]
#### **Problem: Calculate Sigmoid Activation for x = -5 and x = 5**

Using the formula:

f(−5)=1/1+e^-5≈0.0067    
f(5)=1/1+e^+5≈0.9933    

---

### **Softmax Activation Function**

- Converts raw scores (logits) into **probabilities**.
- Outputs a **vector** where each element represents the probability of a class.
- Ensures that **all probabilities sum to 1**.

#### **Formula:**

![[Week_01-02.pptx-1.pdf#page=35&rect=400,286,589,385|Week_01-02.pptx-1, p.35]]
![[Week_01-02.pptx-1.pdf#page=35&rect=314,71,604,224|Week_01-02.pptx-1, p.35]]
#### **Softmax Properties:**

- Used in **multi-class classification** problems.
- The class with the **highest probability** is chosen as the predicted class.


#### **Steps to Calculate Softmax**

1. **Compute exponentials** of each input value.
2. **Sum all exponentials** to get the denominator.
3. **Divide each exponential by the sum** to get the final probabilities.
4. **Ensure values sum to 1** (probability distribution).

#### **Key Takeaways**

- The largest value (logit 5.1) has the highest probability (0.90).
- Smaller values contribute smaller probabilities.
- The softmax function **amplifies differences** between values.

<img width="258" alt="Pasted image 20250312200344" src="https://github.com/user-attachments/assets/bee59244-5b61-4b58-9177-26a7bf29c8e8" />
<img width="250" alt="Pasted image 20250312200418" src="https://github.com/user-attachments/assets/f551703c-3bf4-40e5-bd6a-bad7e8167727" />


-------


### **ReLU (Rectified Linear Unit) Activation Function**

#### **Formula:**

f(x)=max⁡(0,x) 

- Outputs **x** if x>0 otherwise **0**.
- The most widely used activation function in deep learning.
- Helps solve the **vanishing gradient problem** seen in Sigmoid and Tanh.

---

## Backpropagation in Neural Networks

### **Definition**

- **Backpropagation** is the core technique used to train artificial neural networks.
- It **adjusts the weights** of neurons based on the error rate (loss) from previous iterations (epochs).
- The goal is to **minimize the loss function** by improving predictions.

### **How Backpropagation Works**

![[Week_01-02.pptx-1.pdf#page=37&rect=245,78,692,360|Week_01-02.pptx-1, p.37]]
26. **Forward Propagation:**
    
    - Inputs pass through the network layer by layer.
    - The output is compared to the actual label, and an error (loss) is calculated.
27. **Error Calculation:**
    
    - The difference between the predicted output and the actual output is measured using a loss function (e.g., Mean Squared Error, Cross-Entropy Loss).
28. **Backward Propagation:**
    
    - The **gradient** of the loss function is computed with respect to each weight.
    - This is done using the **chain rule of calculus**.
    - The gradients help determine how much to adjust each weight to reduce the loss.
29. **Weight Updates:**
    
    - Weights are updated using **Gradient Descent** (or variants like Adam, RMSprop).
    - This process continues until the model reaches optimal performance.

### **Key Components of Backpropagation**

- **Loss Function:** Measures how far the model’s prediction is from the true output.
- **Gradient Descent Algorithm:** Optimizes weights based on gradients.
- **Learning Rate:** Controls the step size of weight updates.

---




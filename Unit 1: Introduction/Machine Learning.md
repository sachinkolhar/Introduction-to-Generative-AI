# Map
- Machine Learning
	- branchs
		- Supervised Learning : learns to **map inputs to outputs**
			- **Why Use:** predictions, regression , classification, Build models, learn relationships
			- Types 
				- Regression: Predicting continuous values
				- Classification: Predicting discrete labels
		- Unsupervised Learning :
			- finds patterns in **unlabeled data** without predefined outputs.
			- Why Use: discover hidden patterns, group similar data, reduce the dimensionality, detect anomalies and outliers
			- Types
				- **Clustering** (Grouping data based on similarity)
				- **Dimensionality Reduction** (Reducing input features while retaining key information)
		- Reinforcement Learning: learns from rewards or penalties,
			- Why Use : dynamic environments, large action spaces, delayed rewards, Continuously improves, **exploration**, **exploitation**
- Deep Learning
	- Introduction
	- Neural Networks and Artificial Neural Networks
		- The Human Brain and Neurons
		- Artificial Neural Network (ANN)
	- Key Components of ANN Architecture
		- **Nodes**: mathematical functions that process inputs and produce outputs.
		- ** Connections: weights:** determine the strength of the connection between nodes.
		- **Input Layer**: Receives raw input data
		- **Hidden Layers**: feature extraction and transformation.
		- **Output Layer**: final result
		- **Biases**:  model to shift the activation function for better fit.
		- weighted sums :
		- **Activation Functions**:: non-linearity to the model.
			- Common functions: ReLU, Sigmoid, Tanh, Softmax. 
		- **Loss Function**: 
			- difference of  predicted and actual outputs.
			- Mean Squared Error, Cross-Entropy Loss).
		- **Optimization Algorithm**: 
			- Adjusts weights and biases to minimize the loss function.
			- Examples: Gradient Descent, Adam, RMSprop.
	- Activation Functions 
		1. **Sigmoid** (for binary classification) <font color="#ffff00">f(x)=1/1+e^-x     </font> 
		2. **Softmax** (for multi-class classification) : scales numbers into probabilities.
		3. **ReLU (Rectified Linear Unit)** (default for most deep learning tasks)
	- Backpropagation in Neural Networks: 
		- **adjusts the weights** of neurons based on the error rate (loss) from previous iterations (epochs).to **minimize the loss function** by improving predictions.
		- **Forward Propagation:**
		- Error Calculation:  Mean Squared Error, Cross-Entropy Loss).
		- Backward Propagation: gradient of the loss function is computed with respect to each weight.
		- **Weight Updates:**  using **Gradient Descent** (or variants like Adam, RMSprop).
- Generative AI (Gen AI)


-------------
# Machine Learning

![[Week_01-02.pptx-1.pdf#page=10&rect=240,73,719,448|Week_01-02.pptx-1, p.10]]

![[Week_01-02.pptx-1.pdf#page=11&rect=162,79,706,486|Week_01-02.pptx-1, p.11]]


## **Supervised Learning**

![[Week_01-02.pptx-1.pdf#page=14&rect=27,112,653,472|Week_01-02.pptx-1, p.14]]
### **Definition**

- The algorithm is trained using a **labeled dataset**, meaning each input has a corresponding known output.
- The model learns to **map inputs to outputs** and can generalize to make predictions on new data.

### **Why Use Supervised Learning?**

- To **make predictions** on unseen data.
- Build models that can generalize to unseen data
- To **classify data** into predefined categories.
- To **learn relationships** between input and output variables.
- To **generalize models** to work on new data.
- To **perform regression and classification tasks**.

### **Types of Supervised Learning**

1. **Regression** (Predicting continuous values)
    
    - Used when the output variable is continuous.
    - **Examples:**
        - Predicting **stock prices**.
        - Forecasting **sales revenue**.
        - Estimating **house prices**.
2. **Classification** (Predicting discrete labels)
    
    - Used when the output belongs to a specific category.
    - **Examples:**
        - **Face recognition** – Identifying people in images.
        - **Handwriting recognition** – Identifying letters or numbers.
        - **Spam detection** – Classifying emails as spam or not spam.

---

## **Unsupervised Learning**

![[Week_01-02.pptx-1.pdf#page=17&rect=189,159,721,470|Week_01-02.pptx-1, p.17]]
### **Definition**

- The algorithm finds patterns in **unlabeled data** without predefined outputs.
- Can be compared to how humans learn from experience.

### **Why Use Unsupervised Learning?**

- To **discover hidden patterns** in data.
- To **group similar data points** into clusters (**clustering**).
- To **reduce the dimensionality** of data for efficient processing.
- To **learn meaningful representations** of data.
- To **detect anomalies and outliers**.

### **Types of Unsupervised Learning**

1. **Clustering** (Grouping data based on similarity)
    
    - **Example:**
        - **Customer segmentation** – Grouping customers based on behavior.
2. **Dimensionality Reduction** (Reducing input features while retaining key information)
    
    - **Example:**
        - **Visualizing high-dimensional data** in lower dimensions.

---
![[Week_01-02.pptx-1.pdf#page=22&rect=9,126,949,403|Week_01-02.pptx-1, p.22]]
## **Reinforcement Learning (RL)**

![[Week_01-02.pptx-1.pdf#page=19&rect=226,102,732,395|Week_01-02.pptx-1, p.19]]
### **Definition**

- The algorithm **interacts with an environment**, learns from rewards or penalties, and **optimizes long-term performance**.

### **Why Use Reinforcement Learning?**

- Effective in **dynamic environments**.
- Handles **large action spaces** efficiently.
- Deals with **delayed rewards** (outcome is not immediate).
- **Continuously improves** by interacting with the environment.
- Balances **exploration** (trying new actions) and **exploitation** (using known strategies).

### **Key Applications of RL**

1. **Autonomous Driving**
    - Trajectory optimization.
    - Motion planning.
    - Dynamic pathing.
    - Scenario-based learning for highways.
2. **Robotics Control**
    
    - Training robots to perform specific tasks.
3. **Game Playing**
    
    - AI systems like **AlphaGo** and **Deep Q-Networks** learn to play complex games.
4. **Autonomous Helicopter**
    
    - Controlling **drones and helicopters** in real-world conditions.
5. **Animal Training Analogy**
    
    - Example: **Training a puppy** by rewarding good behavior and discouraging bad behavior.

---

## **Comparison: Supervised vs. Unsupervised vs. Reinforcement Learning**

Here is a table summarizing the differences between **Supervised Learning, Unsupervised Learning, and Reinforcement Learning**:

|Feature|**Supervised Learning**|**Unsupervised Learning**|**Reinforcement Learning**|
|---|---|---|---|
|**Definition**|Learns from labeled data (input-output pairs).|Learns from unlabeled data by finding patterns.|Learns by interacting with an environment and receiving rewards/punishments.|
|**Data Type**|Labeled data (with known outputs).|Unlabeled data (no predefined labels).|No predefined dataset; learns from experience.|
|**Goal**|Predict outcomes (classification/regression).|Discover hidden patterns (clustering/dimensionality reduction).|Maximize cumulative rewards over time.|
|**Types**|Regression, Classification|Clustering, Dimensionality Reduction|Model-Free RL, Model-Based RL|
|**Key Techniques**|Decision Trees, Neural Networks, SVM, Naïve Bayes, k-NN|k-Means, Hierarchical Clustering, PCA, Autoencoders|Q-Learning, Deep Q-Networks (DQN), Policy Gradient Methods|
|**Examples**|Spam detection, Fraud detection, House price prediction|Customer segmentation, Anomaly detection, Topic modeling|Autonomous driving, Robotics, Game playing (AlphaGo)|
|**Human Analogy**|A teacher provides correct answers.|A student groups similar objects without guidance.|A child learns to ride a bike through trial and error.|
|**Feedback Type**|Direct supervision with labeled data.|No direct supervision, learns from structure in data.|Reward-based learning through interaction.|


---


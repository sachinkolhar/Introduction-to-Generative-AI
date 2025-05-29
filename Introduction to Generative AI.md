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
		- **Reinforcement Learning:** learns from rewards or penalties,
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
			- <font color="#ffff00">Adjusts weights and biases</font> to minimize the loss function.
			- Examples: Gradient Descent, Adam, RMSprop.
	- Activation Functions 
		1. **Sigmoid** (for binary classification) <font color="#ffff00">f(x)=1/1+e^-x     </font> 
		2. **Softmax** (for multi-class classification) : scales numbers into <font color="#ffff00">probabilities</font>.
		3. **ReLU (Rectified Linear Unit)** (default for most deep learning tasks)
	- Backpropagation in Neural Networks: 
		- **adjusts the weights** of neurons based on the error rate (loss) from previous iterations (epochs).to **minimize the loss function** by improving predictions.
		- **Forward Propagation:**
		- Error Calculation:  Mean Squared Error, Cross-Entropy Loss).
		- Backward Propagation: gradient of the loss function is computed with respect to each weight.
		- **Weight Updates:**  using **Gradient Descent** (or variants like Adam, RMSprop).
- Generative AI (Gen AI)
	- models that can **generate text, images, audio, and video** based on training data.


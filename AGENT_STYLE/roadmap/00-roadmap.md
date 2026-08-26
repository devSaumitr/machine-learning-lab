# ML → DL → GenAI → Agentic AI → MLOps ROADMAP

> Master roadmap — source of truth for topic order and phase structure.
> Progress is tracked in `../progress/current-state.md`, not in this file.

GOAL:
Become job/interview-ready in Machine Learning, Deep Learning, Generative AI, Agentic AI, MLOps and ML System Design by around March, while keeping the roadmap flexible for college and semester exams.

==================================================
PHASE 0 — PYTHON FOUNDATION
==================================================

1. Python Core
[ ] Variables & Data Types
[ ] Input / Output
[ ] Operators
[ ] Conditions
[ ] Loops
[ ] Strings
[ ] Lists
[ ] Tuples
[ ] Sets
[ ] Dictionaries
[ ] Comprehensions
[ ] Functions
[ ] Parameters vs Arguments
[ ] return vs print
[ ] Default Arguments
[ ] Keyword Arguments
[ ] *args
[ ] **kwargs
[ ] Packing & Unpacking
[ ] Scope
[ ] LEGB
[ ] Lambda
[ ] map()
[ ] filter()
[ ] reduce()
[ ] Recursion
[ ] Iterators
[ ] Generators
[ ] Exceptions
[ ] File Handling
[ ] Modules
[ ] Packages
[ ] import
[ ] __name__
[ ] __main__
[ ] OOP
[ ] Classes & Objects
[ ] Constructors
[ ] Instance/Class/Static Methods
[ ] Inheritance
[ ] Polymorphism
[ ] Encapsulation
[ ] Abstraction
[ ] Magic/Dunder Methods

2. Python Internals
[ ] Objects & References
[ ] Identity vs Equality
[ ] Mutable vs Immutable
[ ] Shallow vs Deep Copy
[ ] Memory Model
[ ] Stack vs Heap
[ ] Garbage Collection
[ ] Function Call Stack
[ ] Namespaces
[ ] Closures
[ ] Context Managers
[ ] Decorators

==================================================
PHASE 1 — NUMPY
==================================================

[ ] NumPy Introduction
[ ] ndarray
[ ] Shape
[ ] ndim
[ ] size
[ ] dtype
[ ] Creating Arrays
[ ] Indexing
[ ] Slicing
[ ] Boolean Indexing
[ ] Fancy Indexing
[ ] Reshaping
[ ] Flattening
[ ] Transpose
[ ] Axis
[ ] Aggregations
[ ] Mathematical Operations
[ ] Random Module
[ ] Vectorization
[ ] Broadcasting
[ ] Views vs Copies
[ ] Memory/Layout Basics
[ ] Numerical Stability Basics

ML CONNECTION:
[ ] Understand how ML data becomes numerical arrays
[ ] Understand samples/features
[ ] Understand shapes
[ ] Understand vectorized computation

==================================================
PHASE 2 — PANDAS
==================================================

[ ] Series
[ ] DataFrame
[ ] Index
[ ] Columns
[ ] Selecting Data
[ ] loc
[ ] iloc
[ ] Boolean Filtering
[ ] Sorting
[ ] Missing Values
[ ] Duplicates
[ ] Data Types
[ ] Type Conversion
[ ] apply()
[ ] map()
[ ] GroupBy
[ ] Aggregation
[ ] Merge
[ ] Join
[ ] Concatenation
[ ] Pivot Tables
[ ] Reshaping
[ ] Date/Time Data
[ ] String Operations
[ ] Feature Creation
[ ] Data Cleaning
[ ] Exploratory Data Analysis

ML CONNECTION:
[ ] Raw Data → Clean Data
[ ] Feature Engineering
[ ] Train/Test Preparation
[ ] Data Leakage Awareness

==================================================
PHASE 3 — DATA VISUALIZATION
==================================================

[ ] Matplotlib
[ ] Figure/Axes
[ ] Line Plot
[ ] Bar Plot
[ ] Histogram
[ ] Scatter Plot
[ ] Box Plot
[ ] Subplots
[ ] Labels/Legends
[ ] Visualization Best Practices

[ ] Seaborn
[ ] Distribution Plots
[ ] Categorical Plots
[ ] Correlation Heatmaps
[ ] Pairplots

==================================================
PHASE 4 — MATHEMATICS + STATISTICS
==================================================

1. Statistics
[ ] Mean
[ ] Median
[ ] Mode
[ ] Variance
[ ] Standard Deviation
[ ] Percentiles
[ ] Quartiles
[ ] IQR
[ ] Outliers
[ ] Probability Basics
[ ] Conditional Probability
[ ] Bayes Theorem
[ ] Random Variables
[ ] Distributions
[ ] Normal Distribution
[ ] Binomial Distribution
[ ] Sampling
[ ] Central Limit Theorem
[ ] Confidence Intervals
[ ] Hypothesis Testing
[ ] p-value
[ ] Type I / Type II Errors

2. Linear Algebra
[ ] Scalars
[ ] Vectors
[ ] Matrices
[ ] Matrix Operations
[ ] Dot Product
[ ] Matrix Multiplication
[ ] Transpose
[ ] Inverse
[ ] Rank
[ ] Linear Independence
[ ] Eigenvalues
[ ] Eigenvectors
[ ] Geometric Interpretation

3. Calculus
[ ] Functions
[ ] Limits
[ ] Derivatives
[ ] Partial Derivatives
[ ] Chain Rule
[ ] Gradients
[ ] Gradient Descent

==================================================
PHASE 5 — MACHINE LEARNING FOUNDATIONS
==================================================

[ ] What is Machine Learning?
[ ] AI vs ML vs DL
[ ] Supervised Learning
[ ] Unsupervised Learning
[ ] Semi-Supervised Learning
[ ] Reinforcement Learning Basics

[ ] Dataset
[ ] Features
[ ] Target
[ ] Samples
[ ] Training
[ ] Validation
[ ] Testing
[ ] Generalization
[ ] Overfitting
[ ] Underfitting
[ ] Bias
[ ] Variance
[ ] Bias-Variance Tradeoff

==================================================
PHASE 6 — SCIKIT-LEARN + ML WORKFLOW
==================================================

[ ] Scikit-learn Basics
[ ] Estimator API
[ ] fit()
[ ] predict()
[ ] transform()
[ ] fit_transform()

ML PIPELINE:

Raw Data
↓
EDA
↓
Cleaning
↓
Feature Engineering
↓
Train/Test Split
↓
Preprocessing
↓
Model
↓
Training
↓
Evaluation
↓
Hyperparameter Tuning
↓
Final Model
↓
Deployment

[ ] Train/Test Split
[ ] Cross Validation
[ ] Pipelines
[ ] ColumnTransformer
[ ] Feature Scaling
[ ] Standardization
[ ] Normalization
[ ] Encoding
[ ] One-Hot Encoding
[ ] Label Encoding
[ ] Missing Value Imputation
[ ] Feature Selection
[ ] Feature Engineering
[ ] Data Leakage
[ ] Hyperparameter Tuning
[ ] Grid Search
[ ] Random Search

==================================================
PHASE 7 — SUPERVISED MACHINE LEARNING
==================================================

1. Regression
[ ] Linear Regression
[ ] Multiple Linear Regression
[ ] Polynomial Regression
[ ] Ridge Regression
[ ] Lasso Regression
[ ] Elastic Net

2. Classification
[ ] Logistic Regression
[ ] KNN
[ ] Naive Bayes
[ ] Decision Trees
[ ] Random Forest
[ ] Gradient Boosting
[ ] AdaBoost
[ ] XGBoost
[ ] LightGBM / CatBoost

3. SVM
[ ] Linear SVM
[ ] Kernel Trick
[ ] RBF Kernel

For every major algorithm:

[ ] Intuition
[ ] Mathematics
[ ] Algorithm
[ ] Implementation
[ ] Hyperparameters
[ ] Advantages
[ ] Disadvantages
[ ] Failure Cases
[ ] When to Use
[ ] Interview Questions

==================================================
PHASE 8 — UNSUPERVISED LEARNING
==================================================

[ ] Clustering
[ ] K-Means
[ ] Hierarchical Clustering
[ ] DBSCAN
[ ] Gaussian Mixture Models

[ ] Dimensionality Reduction
[ ] PCA
[ ] t-SNE
[ ] UMAP

[ ] Anomaly Detection Basics

==================================================
PHASE 9 — MODEL EVALUATION
==================================================

Regression:
[ ] MAE
[ ] MSE
[ ] RMSE
[ ] R²
[ ] Adjusted R²

Classification:
[ ] Confusion Matrix
[ ] Accuracy
[ ] Precision
[ ] Recall
[ ] F1 Score
[ ] Specificity
[ ] ROC
[ ] AUC
[ ] Precision-Recall Curve

[ ] Cross Validation
[ ] Stratified Cross Validation
[ ] Imbalanced Data
[ ] Class Weights
[ ] Sampling Techniques
[ ] Threshold Tuning

==================================================
PHASE 10 — FEATURE ENGINEERING
==================================================

[ ] Numerical Features
[ ] Categorical Features
[ ] Missing Values
[ ] Outliers
[ ] Scaling
[ ] Encoding
[ ] Binning
[ ] Log Transformation
[ ] Interaction Features
[ ] Polynomial Features
[ ] Feature Selection
[ ] Dimensionality Reduction
[ ] Leakage Prevention

==================================================
PHASE 11 — ML PROJECTS
==================================================

Project 1:
[ ] Complete End-to-End Regression Project

Project 2:
[ ] Complete Classification Project

Project 3:
[ ] Complete Unsupervised Learning Project

Project 4:
[ ] Real-World End-to-End ML Project

Every project:

[ ] Data Collection
[ ] Data Cleaning
[ ] EDA
[ ] Feature Engineering
[ ] Model Selection
[ ] Training
[ ] Evaluation
[ ] Hyperparameter Tuning
[ ] Pipeline
[ ] API
[ ] Deployment
[ ] Documentation
[ ] GitHub

==================================================
PHASE 12 — DEEP LEARNING FOUNDATIONS
==================================================

[ ] Neural Network Intuition
[ ] Perceptron
[ ] Neuron
[ ] Layers
[ ] Weights
[ ] Bias
[ ] Activation Functions
[ ] Sigmoid
[ ] Tanh
[ ] ReLU
[ ] Softmax

[ ] Forward Propagation
[ ] Loss Functions
[ ] Backpropagation
[ ] Gradients
[ ] Gradient Descent
[ ] Learning Rate
[ ] Optimizers
[ ] SGD
[ ] Momentum
[ ] Adam
[ ] AdamW

[ ] Batch
[ ] Epoch
[ ] Iteration
[ ] Batch Size

[ ] Overfitting in Neural Networks
[ ] Dropout
[ ] Regularization
[ ] Batch Normalization
[ ] Early Stopping

==================================================
PHASE 13 — PYTORCH
==================================================

[ ] PyTorch Basics
[ ] Tensors
[ ] Tensor Shapes
[ ] Tensor Operations
[ ] Autograd
[ ] Computational Graph
[ ] nn.Module
[ ] Dataset
[ ] DataLoader
[ ] Training Loop
[ ] Validation Loop
[ ] Optimizers
[ ] Loss Functions
[ ] GPU / CUDA Basics
[ ] Saving / Loading Models

==================================================
PHASE 14 — COMPUTER VISION
==================================================

[ ] Image Representation
[ ] CNN
[ ] Convolution
[ ] Filters
[ ] Kernels
[ ] Stride
[ ] Padding
[ ] Pooling
[ ] Feature Maps

[ ] CNN Architectures
[ ] LeNet
[ ] AlexNet
[ ] VGG
[ ] ResNet
[ ] Transfer Learning
[ ] Fine-Tuning

[ ] Image Classification
[ ] Object Detection Basics
[ ] Segmentation Basics

==================================================
PHASE 15 — SEQUENCE MODELS
==================================================

[ ] Sequential Data
[ ] RNN
[ ] Vanishing Gradient
[ ] Exploding Gradient
[ ] LSTM
[ ] GRU
[ ] Bidirectional RNN
[ ] Sequence-to-Sequence

==================================================
PHASE 16 — ATTENTION + TRANSFORMERS
==================================================

[ ] Why Attention?
[ ] Query
[ ] Key
[ ] Value
[ ] Attention Scores
[ ] Scaled Dot Product Attention
[ ] Self Attention
[ ] Multi-Head Attention
[ ] Positional Encoding
[ ] Encoder
[ ] Decoder
[ ] Transformer Architecture

[ ] BERT
[ ] GPT
[ ] Encoder vs Decoder vs Encoder-Decoder

==================================================
PHASE 17 — NLP FOUNDATIONS
==================================================

[ ] Text Preprocessing
[ ] Tokenization
[ ] Vocabulary
[ ] Stop Words
[ ] Stemming
[ ] Lemmatization
[ ] Bag of Words
[ ] TF-IDF
[ ] Word Embeddings
[ ] Word2Vec
[ ] GloVe
[ ] Contextual Embeddings

==================================================
PHASE 18 — LARGE LANGUAGE MODELS
==================================================

[ ] What is an LLM?
[ ] Pretraining
[ ] Next Token Prediction
[ ] Tokenization
[ ] Context Window
[ ] Parameters
[ ] Inference
[ ] Temperature
[ ] Top-k
[ ] Top-p

[ ] Prompt Engineering
[ ] System Prompt
[ ] User Prompt
[ ] Few-Shot Prompting
[ ] Chain-of-Thought Concepts
[ ] Structured Outputs

==================================================
PHASE 19 — GENERATIVE AI
==================================================

[ ] Generative AI Fundamentals
[ ] LLM APIs
[ ] Chat Models
[ ] Prompt Templates
[ ] Output Parsers
[ ] Structured Generation
[ ] Function / Tool Calling

[ ] Embeddings
[ ] Vector Representations
[ ] Similarity
[ ] Cosine Similarity

==================================================
PHASE 20 — LANGCHAIN
==================================================

[ ] LangChain Fundamentals
[ ] Models
[ ] Prompts
[ ] Output Parsers
[ ] Runnables
[ ] LCEL
[ ] Chains
[ ] Components
[ ] Document Loaders
[ ] Text Splitters
[ ] Embeddings
[ ] Vector Stores
[ ] Retrievers
[ ] Memory
[ ] Tool Calling

==================================================
PHASE 21 — RAG
==================================================

[ ] What is RAG?
[ ] Why RAG?
[ ] RAG Architecture

Documents
↓
Loading
↓
Chunking
↓
Embedding
↓
Vector Database
↓
Retrieval
↓
Context
↓
LLM
↓
Answer

[ ] Chunking Strategies
[ ] Embedding Models
[ ] Vector Databases
[ ] Similarity Search
[ ] Metadata Filtering
[ ] Hybrid Search
[ ] Reranking
[ ] Query Transformation
[ ] Retrieval Evaluation
[ ] RAG Evaluation
[ ] Hallucination
[ ] Context Quality

==================================================
PHASE 22 — FINE-TUNING LLMs
==================================================

[ ] Pretraining vs Fine-Tuning
[ ] Instruction Tuning
[ ] Supervised Fine-Tuning
[ ] PEFT
[ ] LoRA
[ ] QLoRA
[ ] Dataset Preparation
[ ] Training Configuration
[ ] Evaluation
[ ] Quantization
[ ] Model Selection
[ ] Fine-Tuning vs RAG

==================================================
PHASE 23 — AGENTIC AI
==================================================

[ ] What is an AI Agent?
[ ] Agent Architecture
[ ] Perception
[ ] Reasoning
[ ] Planning
[ ] Action

[ ] Tool Calling
[ ] Function Calling
[ ] Memory
[ ] Planning
[ ] Reflection
[ ] Agent Loops
[ ] State
[ ] Agent Evaluation
[ ] Failure Modes

[ ] Single-Agent Systems
[ ] Multi-Agent Systems
[ ] Agent Collaboration

==================================================
PHASE 24 — LANGGRAPH / AGENT FRAMEWORKS
==================================================

[ ] Graph-Based Agents
[ ] Nodes
[ ] Edges
[ ] State
[ ] Checkpoints
[ ] Conditional Routing
[ ] Human-in-the-Loop
[ ] Tool Nodes
[ ] Multi-Agent Graphs
[ ] Agent Persistence
[ ] Agent Evaluation

==================================================
PHASE 25 — ML ENGINEERING
==================================================

[ ] Project Structure
[ ] Configuration Management
[ ] Logging
[ ] Testing
[ ] Unit Tests
[ ] Integration Tests
[ ] Type Hints
[ ] Environment Management
[ ] Dependency Management
[ ] Git
[ ] GitHub
[ ] Documentation
[ ] Clean Code

==================================================
PHASE 26 — DEPLOYMENT
==================================================

[ ] FastAPI
[ ] REST APIs
[ ] Request/Response Models
[ ] Model Serving
[ ] Docker
[ ] Dockerfile
[ ] Docker Compose
[ ] Environment Variables
[ ] Production Configuration

==================================================
PHASE 27 — MLOPS
==================================================

[ ] ML Lifecycle
[ ] Experiment Tracking
[ ] MLflow
[ ] Model Registry
[ ] Data Versioning
[ ] Model Versioning
[ ] CI/CD
[ ] Automated Testing
[ ] Model Deployment
[ ] Monitoring
[ ] Data Drift
[ ] Model Drift
[ ] Logging
[ ] Observability

==================================================
PHASE 28 — LLMOPS
==================================================

[ ] LLM Evaluation
[ ] Prompt Versioning
[ ] Model Versioning
[ ] Token Cost
[ ] Latency
[ ] Caching
[ ] Guardrails
[ ] Observability
[ ] RAG Monitoring
[ ] Agent Monitoring
[ ] Production Failure Handling

==================================================
PHASE 29 — CLOUD
==================================================

[ ] Cloud Fundamentals
[ ] Compute
[ ] Storage
[ ] Databases
[ ] Networking Basics
[ ] Containers
[ ] Serverless Basics

Choose one primary cloud:

[ ] AWS
OR
[ ] Azure
OR
[ ] GCP

Then learn:

[ ] Deployment
[ ] Storage
[ ] Compute
[ ] Databases
[ ] IAM
[ ] Monitoring

==================================================
PHASE 30 — ML SYSTEM DESIGN
==================================================

[ ] ML System Design Fundamentals
[ ] Requirements
[ ] Data Collection
[ ] Data Processing
[ ] Feature Engineering
[ ] Feature Store
[ ] Training Pipeline
[ ] Model Serving
[ ] Batch Inference
[ ] Online Inference
[ ] Scaling
[ ] Caching
[ ] Monitoring
[ ] Failure Handling
[ ] Tradeoffs
[ ] Cost
[ ] Latency
[ ] Reliability

Design systems such as:

[ ] Recommendation System
[ ] Search System
[ ] Fraud Detection
[ ] Spam Detection
[ ] Ranking System
[ ] Image Classification System
[ ] LLM Application
[ ] RAG System
[ ] AI Agent

==================================================
PHASE 31 — PROJECT PORTFOLIO
==================================================

Build progressively stronger projects.

LEVEL 1:
[ ] Python Project

LEVEL 2:
[ ] Data Analysis Project

LEVEL 3:
[ ] Classical ML Project

LEVEL 4:
[ ] End-to-End ML Deployment Project

LEVEL 5:
[ ] Deep Learning Project

LEVEL 6:
[ ] Computer Vision / NLP Project

LEVEL 7:
[ ] GenAI Application

LEVEL 8:
[ ] Production RAG Application

LEVEL 9:
[ ] Agentic AI Application

LEVEL 10:
[ ] Full Production-Style AI System

Every major project should have:

[ ] GitHub Repository
[ ] README
[ ] Architecture Diagram
[ ] Dataset / Data Pipeline
[ ] Training
[ ] Evaluation
[ ] API
[ ] Deployment
[ ] Testing
[ ] Monitoring
[ ] Documentation

==================================================
PHASE 32 — INTERVIEW PREPARATION
==================================================

Python:
[ ] Python Fundamentals
[ ] OOP
[ ] Python Internals
[ ] Data Structures
[ ] Functions
[ ] Iterators / Generators

DSA:
[ ] Arrays
[ ] Strings
[ ] Linked Lists
[ ] Stack
[ ] Queue
[ ] Binary Tree
[ ] BST
[ ] Heap
[ ] Graph
[ ] Recursion
[ ] Backtracking
[ ] Greedy
[ ] Dynamic Programming
[ ] Binary Search
[ ] Sliding Window
[ ] Two Pointers
[ ] Prefix/Suffix

ML:
[ ] ML Fundamentals
[ ] Algorithms
[ ] Evaluation
[ ] Feature Engineering
[ ] Bias/Variance
[ ] Overfitting
[ ] Data Leakage
[ ] Model Selection

DL:
[ ] Neural Networks
[ ] Backpropagation
[ ] Optimizers
[ ] CNN
[ ] RNN
[ ] Transformers

GenAI:
[ ] LLMs
[ ] Embeddings
[ ] RAG
[ ] Fine-Tuning
[ ] Agents

MLOps:
[ ] Deployment
[ ] Docker
[ ] MLflow
[ ] Monitoring
[ ] CI/CD

System Design:
[ ] ML System Design
[ ] LLM System Design
[ ] RAG System Design
[ ] Agent System Design

==================================================
FINAL TARGET
==================================================

By the end of the roadmap, I should be able to:

[ ] Write strong Python
[ ] Understand Python internals
[ ] Work confidently with NumPy
[ ] Work confidently with Pandas
[ ] Perform EDA
[ ] Understand statistics and linear algebra required for ML
[ ] Build ML models
[ ] Evaluate models correctly
[ ] Engineer features
[ ] Build end-to-end ML projects
[ ] Understand deep learning
[ ] Build models with PyTorch
[ ] Understand CNNs/RNNs/Transformers
[ ] Understand LLMs
[ ] Build GenAI applications
[ ] Build RAG systems
[ ] Fine-tune models
[ ] Build AI agents
[ ] Build multi-agent systems
[ ] Deploy ML/AI systems
[ ] Understand MLOps
[ ] Monitor production models
[ ] Design ML/LLM systems
[ ] Build a strong project portfolio
[ ] Defend technical decisions in interviews

CORE PRINCIPLE:

Understand
↓
Implement
↓
Debug
↓
Apply
↓
Build
↓
Deploy
↓
Explain
↓
Defend in Interview

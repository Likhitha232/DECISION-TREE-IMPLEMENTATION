# DECISION-TREE-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: H LIKITHA

*INTERN ID*: CTO6DF2216

*DOMAIN*: MACHINE LEARNING

*DURATION*: 6 WEEKS

*MENTOR*: NEELA SANTOSH

This project's implementation of Decision Trees is a basic yet effective illustration of supervised machine learning with Python and Scikit-learn.  Classifying iris flower species using a decision tree model and visualizing the model's decision-making structure are the main objectives.  The famous Iris dataset, a common benchmark dataset in machine learning, was used for this assignment. It comprises 150 examples of iris blooms divided into three classes: Virginica, Versicolor, and Setosa.  Four characteristics are used to describe each flower: petal length, petal breadth, sepal length, and sepal width.  The first step in the implementation is importing the necessary Python libraries: Scikit-learn for machine learning operations like dataset loading, model training, evaluation, and visualization; Pandas and NumPy for data manipulation; and Matplotlib and Seaborn for visualization. The development environment utilized is Jupyter Notebook, an open-source web application that is perfect for data science and machine learning processes since it allows live code, equations, and visualizations.

 The load_iris() method from Scikit-learn is used to load the dataset as the initial stage in the procedure.  After that, the data is transformed into a Pandas DataFrame for simple editing and viewing.  To transfer the numeric class labels to the names of the corresponding flower species, a new column is added to the DataFrame.  Dividing the dataset into training and testing sets is the next important step.  The train_test_split() function is used to accomplish this, using 30% of the data for testing and 70% for training.This guarantees that the model is tested on unseen data to determine its capacity for generalization and is trained on a portion of the data.  Following data separation, DecisionTreeClassifier is used to instantiate a Decision Tree model.  Crucial options include max_depth=3, which restricts the tree's depth to prevent overfitting and preserve interpretability, and criterion='entropy', which divides the tree nodes using information gain.

 The next step is to display the decision tree after the model has been trained on the training data using the.fit() method.  The plot_tree() method from the tree module of Scikit-learn is used to accomplish this.  The full tree structure is shown in the visualization, together with feature splits, class labels, entropy or Gini values, and sample counts at every node.giving a clear picture of the model's decision-making process.  The.predict() function is then used to make predictions on the test set in order to assess the trained model.  Using accuracy_score() and classification_report(), assessment metrics including accuracy, precision, recall, and F1-score are calculated by comparing these predictions to the actual class labels.  The model performs remarkably well on this dataset, as evidenced by its overall accuracy of about 97.78%.  All three classes have almost flawless scores in the classification report, and the right and wrong classifications are graphically represented by a confusion matrix generated using Seaborn's heatmap().  The matrix shows low levels of misclassification, indicating the model's efficacy. 
 From data loading and preprocessing to model training, visualization, and evaluation, this project demonstrates a comprehensive workflow for developing a classification model using decision trees.  Because of their interpretability, simplicity of usage, and capacity to handle both numerical and categorical data, decision trees are particularly beneficial.  They serve as the foundation for more complex ensemble techniques like Gradient Boosting and Random Forest.  This implementation's ease of use and the robust visualization and assessment capabilities offered by Python modules make it an excellent place for novices to start when learning machine learning.  Additionally, code, graphs, and explanations can all be seamlessly integrated with the Jupyter Notebook platform, which makes the workflow both effective and instructive.

 *OUTPUT*: 

 ![Image](https://github.com/user-attachments/assets/b0ce10fa-4687-4a3b-a7c5-ffca5ce712bb)

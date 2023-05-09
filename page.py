import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, roc_curve,RocCurveDisplay,precision_recall_curve

# Define the list of available algorithms
ALGORITHMS = ['Decision Tree', 'Logistic Regression', 'KNN']
METRICS=['Plot_Confusion_Matrix','ROC_Curve','Precision_Recall_Curve']

# Define a function for each algorithm

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split, max_depth, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    
    def accuracy(self,y_true,y_pred):
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        acc = correct / total
        return acc
    
    def precision(self,y_true, y_pred,positive_class=1):
        true_positives = np.sum((y_true == positive_class) & (y_pred == positive_class))
        predicted_positives = np.sum(y_pred ==positive_class)
        if predicted_positives == 0:
            return 0.0
        prec = true_positives / predicted_positives
        return prec
        
    def calculate_recall(self,y_true, y_pred):
        # Calculate true positives and false negatives
        true_positives = np.logical_and(y_true == 1, y_pred == 1).sum()
        false_negatives = np.logical_and(y_true == 1, y_pred == 0).sum()

        # Calculate recall
        recall = true_positives / (true_positives + false_negatives)
        
        return recall
    
    def f1_score(self,y_true, y_pred):

        prec=self.precision(y_true,y_pred)
        rec=self.calculate_recall(y_true,y_pred)

        f1_score = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

        return f1_score

class LogisticRegression:
    def __init__(self, lr, n_iters):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize the parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Perform gradient descent
        for i in range(self.n_iters):
            # Compute the predicted values
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
            
            # Compute the gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update the parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, x):
        # Compute the predicted values
        y_pred = self.sigmoid(np.dot(x, self.weights) + self.bias)
        
        # Convert the predicted values to binary labels
        return np.round(y_pred).astype(int)
    
    
    def accuracy(self,y_true,y_pred):
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        acc = correct / total
        return acc
    
    def precision(self,y_true, y_pred, positive_class=1):
        tp=0
        fp=0
        for i in range(len(y_true)):
            if y_true[i] == positive_class and y_pred[i] == positive_class:
                tp += 1
            elif y_true[i] != positive_class and y_pred[i] == positive_class:
                fp += 1
        prec = tp / (tp + fp)
        return prec
    
    def calculate_recall(self,y_true, y_pred, positive_class=1):
        tp = 0
        fn = 0
        for i in range(len(y_true)):
            if y_true[i] == positive_class and y_pred[i] == positive_class:
                tp += 1
            elif y_true[i] == positive_class and y_pred[i] != positive_class:
                fn += 1
        recall = tp / (tp + fn)

        return recall
    
    def f1_score(self,y_true, y_pred):

        prec=self.precision(y_true,y_pred)
        rec=self.calculate_recall(y_true,y_pred)

        f1_score = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

        return f1_score

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # Calculate distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common_label

        return y_pred

    def accuracy(self,y_true,y_pred):
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        acc = correct / total
        return acc
    
    def precision(self,y_true, y_pred, positive_class=1):
        tp=0
        fp=0
        for i in range(len(y_true)):
            if y_true[i] == positive_class and y_pred[i] == positive_class:
                tp += 1
            elif y_true[i] != positive_class and y_pred[i] == positive_class:
                fp += 1
        prec = tp / (tp + fp)
        return prec
    
    def calculate_recall(self,y_true, y_pred, positive_class=1):
        tp = 0
        fn = 0
        for i in range(len(y_true)):
            if y_true[i] == positive_class and y_pred[i] == positive_class:
                tp += 1
            elif y_true[i] == positive_class and y_pred[i] != positive_class:
                fn += 1
        recall = tp / (tp + fn)

        return recall
    
    def f1_score(self,y_true, y_pred):

        prec=self.precision(y_true,y_pred)
        rec=self.calculate_recall(y_true,y_pred)

        f1_score = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

        return f1_score
    

def app():

    st.title("INFOGRAPH(Machine Learning and Data Visualization Tool)")
    st.write("\n\n\n")
    st.header("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? ")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? ")

    data=pd.read_csv(r"C:\Users\VAMSI8\Dropbox\PC\Downloads\mushroom.csv")

    df=data.head()
    st.table(df)

    subset=data.sample(n=1000,random_state=42)

    x=subset.iloc[:,1:].values
    y=subset.iloc[:,0].values

    def train_test_split(X, y, test_size, random_state):
        if random_state:
            np.random.seed(random_state)
        shuffled_indices = np.random.permutation(len(X))
        test_size = int(len(X) * test_size)
        test_indices = shuffled_indices[:test_size]
        train_indices = shuffled_indices[test_size:]
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test
    
   
    st.sidebar.title("Selecting a Machine Learning Algorithm")
    
    # Display the available algorithms
    st.sidebar.subheader("Select Algorithm")
    algorithm = st.sidebar.selectbox("Choose an algorithm", ALGORITHMS)
    st.sidebar.write(f"You selected: {algorithm}")
    
    # Run the selected algorithm
    if algorithm == 'Decision Tree':
        st.sidebar.subheader("Model Hyperparameters")

        n_estim = st.sidebar.number_input("Minimum number of sample splits", 100, 
500, step=100, key='n_estim')
        
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 10, 
step=1, key='max_depth')
        
       
        st.sidebar.title("choose metrics")
        metrics = st.sidebar.multiselect("what to plot?", METRICS)
        st.sidebar.write(f"You selected: {metrics}")


        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42)

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Decision Tree Results")

            rnd = DecisionTree(min_samples_split=n_estim, max_depth=max_depth)
            rnd.fit(x_train, y_train)
            predictions = rnd.predict(x_test)
            acc=rnd.accuracy(y_test,predictions)
            prec=rnd.precision(y_test,predictions)
            recall=rnd.calculate_recall(y_test,predictions)
            f1_score=rnd.f1_score(y_test,predictions)
            st.write("Accuracy",acc)
            st.write("Precison",prec)
            st.write("Recall",recall)
            st.write("F1_score",f1_score)

            if 'Plot_Confusion_Matrix' in metrics:
                st.markdown("#### confusion  matrix :")
                cnf=confusion_matrix(y_test,predictions)
                mat=ConfusionMatrixDisplay(confusion_matrix=cnf,display_labels=["edible", "poisonous"])
                fig,ax=plt.subplots(figsize=(6, 6))
                mat.plot(ax=ax)
                ax.set_title('Confusion matrix')
                ax.set_xlabel('Predicted label')
                ax.set_ylabel('True label')

                st.pyplot(fig)
                st.write("\n\n")
            
            if 'ROC_Curve' in metrics:

                st.markdown("#### ROC curve :")
                fpr,tpr,thresholds=roc_curve(y_test,predictions)
                mat=RocCurveDisplay(fpr=fpr,tpr=tpr)
                curve,ax=plt.subplots(figsize=(6, 6))
                mat.plot(ax=ax)
                ax.set_title('roc curve')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                
                st.pyplot(curve)
                st.write("\n\n")

            if 'Precision_Recall_Curve' in metrics:

                st.markdown("#### precision recall curve :")
                                
                pre,rec,thresholds=precision_recall_curve(y_test,predictions)
                fig, ax = plt.subplots()
                ax.plot(rec, pre)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                st.pyplot(fig)
                

    if algorithm == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, 
step=0.01, key='C')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 1000, 
key='max_iter')
       
        st.sidebar.title("choose metrics")
        metrics = st.sidebar.multiselect("what to plot?", METRICS)
        st.sidebar.write(f"You selected: {metrics}")

        X=data.drop('class',axis=1).values
        Y=data['class'].values

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            
            X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25, random_state=0)

            lgm = LogisticRegression(n_iters=max_iter,lr=C)
            lgm.fit(X_train, y_train)
            predictions = lgm.predict(X_test)
            acc=lgm.accuracy(y_test,predictions)
            st.write("Accuracy",acc.round(2))
            y_pred=lgm.predict(X_test)
            prec=lgm.precision(y_test,y_pred)
            st.write("Precision",prec)
            predicti=lgm.predict(X_test)
            recall=lgm.calculate_recall(y_test,predicti)
            st.write("Recall",recall)
            f1_score=lgm.f1_score(y_test,y_pred)
            st.write("F1_score",f1_score)
            st.write("\n\n")

            if 'Plot_Confusion_Matrix' in metrics:
                st.markdown("#### confusion  matrix :")
                cnf=confusion_matrix(y_test,y_pred)
                mat=ConfusionMatrixDisplay(confusion_matrix=cnf,display_labels=["edible", "poisonous"])
                fig,ax=plt.subplots(figsize=(6, 6))
                mat.plot(ax=ax)
                ax.set_title('Confusion matrix')
                ax.set_xlabel('Predicted label')
                ax.set_ylabel('True label')

                st.pyplot(fig)
                st.write("\n\n")
            
            if 'ROC_Curve' in metrics:

                st.markdown("#### ROC curve :")
                fpr,tpr,thresholds=roc_curve(y_test,y_pred)
                mat=RocCurveDisplay(fpr=fpr,tpr=tpr)
                curve,ax=plt.subplots(figsize=(6, 6))
                mat.plot(ax=ax)
                ax.set_title('roc curve')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                
                st.pyplot(curve)
                st.write("\n\n")

            if 'Precision_Recall_Curve' in metrics:

                st.markdown("#### precision recall curve :")
                                
                pre,rec,thresholds=precision_recall_curve(y_test,y_pred)
                fig, ax = plt.subplots()
                ax.plot(rec, pre)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                st.pyplot(fig)

    if algorithm == 'KNN':
        st.sidebar.subheader("Model Hyperparameters")
    
        #Lambda_param = st.sidebar.slider("Select regularization parameter (C)", 0.0, 10.0, 1.0)
        n_iters = st.sidebar.slider("Select number of iterations", 3, 20,key='n_iters')

        st.sidebar.title("choose metrics")
        metrics = st.sidebar.multiselect("what to plot?", METRICS)
        st.sidebar.write(f"You selected: {metrics}")

        if st.sidebar.button("Classify", key='classify'):
            st.subheader(" (KNN) Results")

            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)

            knn = KNN(k=n_iters)  
            knn.fit(x_train, y_train)
            predictions=knn.predict(x_test)
            acc=knn.accuracy(y_test,predictions)
            st.write("Accuracy",acc.round(2))
            prec=knn.precision(y_test,predictions)
            st.write("Precision",prec)
            recall=knn.calculate_recall(y_test,predictions)
            st.write("Recall",recall)
            f1_score=knn.f1_score(y_test,predictions)
            st.write("F1_score",f1_score)
            st.write("\n\n")

            if 'Plot_Confusion_Matrix' in metrics:
                st.markdown("#### confusion  matrix :")
                cnf=confusion_matrix(y_test,predictions)
                mat=ConfusionMatrixDisplay(confusion_matrix=cnf,display_labels=["edible", "poisonous"])
                fig,ax=plt.subplots(figsize=(6, 6))
                mat.plot(ax=ax)
                ax.set_title('Confusion matrix')
                ax.set_xlabel('Predicted label')
                ax.set_ylabel('True label')

                st.pyplot(fig)
                st.write("\n\n")
            
            if 'ROC_Curve' in metrics:

                st.markdown("#### ROC curve :")
                fpr,tpr,thresholds=roc_curve(y_test,predictions)
                mat=RocCurveDisplay(fpr=fpr,tpr=tpr)
                curve,ax=plt.subplots(figsize=(6, 6))
                mat.plot(ax=ax)
                ax.set_title('roc curve')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                
                st.pyplot(curve)
                st.write("\n\n")

            if 'Precision_Recall_Curve' in metrics:

                st.markdown("#### precision recall curve :")
                                
                pre,rec,thresholds=precision_recall_curve(y_test,predictions)
                fig, ax = plt.subplots()
                ax.plot(rec, pre)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                st.pyplot(fig)


app()

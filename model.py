import sklearn

random_state = 30
label = "hello"

#We define the labels and features of the dataset and split them accordingly
X_columns = list(filter(lambda x: x != label, df.columns))
y_columns = [label]

#We physically split the data
X_df = df.drop(columns=y_columns)
Y_df = df.drop(columns=x_columns)

#We format the data so it can be used by the model
X = X_df.copy().to_numpy()
y = y_df.copy().to_numpy().reshape((-1,))

#Here we split the data first into 60% training and 40% testing and valitating.
X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.40, random_state=random_state)

#Here we further split the data into a validation and training set
X_test, X_validation, y_test, y_validation = train_test_split(X_hold, y_hold, test_size=0.5, random_state=random_state)

#Here we format the X and y data so we can try and find the optimal weights for our training.
X_grid = np.concatenate((X_train, X_test))
y_grid = np.concatenate((y_train, y_test))

regularization_terms = np.linspace(0.0001, 5, 20)
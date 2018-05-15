import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import StandardScaler
import mlinstrumentation

# init/start instrumentation
mlinstrumentation.start()

# optionally, set seed for consistency - for tutorial purposes
np.random.seed(1333)

# define your features in dataset
features = ['Class', 'Sex', 'Age', 'Fare']

# read your dataset
df = pd.read_csv("data/titanic.csv")

# split train/test data
df_train = df.iloc[:712, :]
df_test  = df.iloc[712:, :]

# pre-process train data
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[features].values)
y_train = df_train['Survived'].values
y_train_onehot = pd.get_dummies(df_train['Survived']).values

# prepare test data
X_test = scaler.transform(df_test[features].values)
y_test = df_test['Survived'].values

# define model
model = Sequential()
model.add(Dense(4, input_shape=(4,)))
model.add(Dense(2))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# train model
model.fit(X_train, y_train_onehot, epochs=1)

# prediction from model
y_prediction = model.predict_classes(X_test)

print("\n\naccuracy", np.sum(y_prediction == y_test) / float(len(y_test)))

# deinit/stop instrumentation
mlinstrumentation.stop()
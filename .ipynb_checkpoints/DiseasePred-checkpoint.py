# import library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train = pd.read_csv('data/Training.csv')
test= pd.read_csv('data/Testing.csv')
train.head()

#########################
#####   Data Prep   #####
#########################

train.shape
train.info()
train.isna().sum() # calculation of the missing value
train.nunique()

X = train.drop('prognosis',axis=1)
#X #X are in DataFrame, its easy to change it into array to match with the target


#########################
##### Split Dataset #####
#########################

X = train.drop('prognosis',axis=1) #X are in DataFrame, its easy to change it into array to match with the target
x = np.array(X) # change X DataFrame into array 

y = np.array(train['prognosis'])
y = pd.get_dummies(y).values

# split data to 80/20 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)


#########################
###  Initialize param ###
#########################

##### Initialize variables
learning_rate = 0.1
iterations = 6000
N = y_train.size

# number of input features
input_size = 57
# number of hidden layers neurons
hidden_size = 50
# number of neurons at the output layer
output_size = 12

results = pd.DataFrame(columns=["mse", "accuracy"])



# Initialize weights
np.random.seed(10)

# initializing weight for the hidden layer
W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))

# initializing weight for the output layer
W2 = np.random.normal(scale=0.5, size=(hidden_size , output_size)) 


#########################
##### Function BPNN #####
#########################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2*y_pred.size)
    
def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()



#########################
##### BPNN Process  #####
#########################

for itr in range(iterations):    
    
    # feedforward propagation
    # on hidden layer
    Z1 = np.dot(X_train, W1)
    A1 = sigmoid(Z1)

    # on output layer
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    
    
    # Calculating error
    mse = mean_squared_error(A2, y_train)
    acc = accuracy(A2, y_train)
    results=results.append({"mse":mse, "accuracy":acc},ignore_index=True )
    
    # backpropagation
    E1 = A2 - y_train
    dW1 = E1 * A2 * (1 - A2)

    E2 = np.dot(dW1, W2.T)
    dW2 = E2 * A1 * (1 - A1)

    
    # weight updates
    W2_update = np.dot(A1.T, dW1) / N
    W1_update = np.dot(X_train.T, dW2) / N

    W2 = W2 - learning_rate * W2_update
    W1 = W1 - learning_rate * W1_update

    
#########################
#######  Result  ########
#########################

results.mse.plot(title="Mean Squared Error")
plt.tight_layout()
plt.savefig("images/mse.png",dpi=120) 
plt.close()

print("Accuracy: {}".format(acc))
results.accuracy.plot(title="Accuracy")

plt.tight_layout()
plt.savefig("images/accuracy.png",dpi=120) 
plt.close()


#########################
######   Testing  #######
#########################

## TEST
# feedforward
Z1 = np.dot(X_test, W1)
A1 = sigmoid(Z1)

Z2 = np.dot(A1, W2)
A2 = sigmoid(Z2)

acc = accuracy(A2, y_test)
print("Accuracy: {}".format(acc))


predictions_df = pd.DataFrame( list(A2.argmax(axis=1)), list(y_test.argmax(axis=1)) , columns=['predicted'] )
predictions_df = predictions_df.reset_index()
predictions_df = predictions_df.rename(columns={'index':'prognosis'})

predictions_df['result'] = ''

for x in range(len(predictions_df)) :
    if predictions_df['prognosis'][x] == predictions_df['predicted'][x] :
        predictions_df['result'].iloc[x]='Correct'
    else :
        predictions_df['result'].iloc[x]='Incorrect'

        
predictions_df

correct = len(predictions_df[predictions_df['result']=='Correct'])
total = len(predictions_df)
print(f'{correct} correct prediction out of {total}')




with open("metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: %2.1f%%\n" % acc)
        outfile.write("Test variance explained: %2.1f%%\n" % mse)
        
        outfile.write(predictions_df)
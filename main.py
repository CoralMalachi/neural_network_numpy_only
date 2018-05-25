import numpy as np


def tanh_deriv_function(x):
    return (1 - np.power(x,2))


def sigmoid(matrix):
    return 1/(1 + np.exp(-matrix))

def sigmoid_derivative(x):
    return (1.0-x)*x


###############################################################
#Function Name: softmax
#Function input: matrix z
#Function output:softmax result
#Function Action: the function calcuslate the result of softmax function
################################################################
def softmax(z):
    # Calculate exponent term first
    mone = np.exp(z)
    return mone/np.sum(mone,axis=1,keepdims=True)

#################################################
#Function name:calc_loss
#Function input: y and y_hat
#Function output: loss
#Function Action:# Loss formula, np.sum
#  sums up the entire matrix and therefore does the
#  job of two sums from the formula
################################################
def calc_loss(y,y_hat):
    loss=np.sum(-y_hat*np.log(y_hat))
    loss=loss/y.shape[0]
    return loss

#################################################
#Function name:loss_softmax
#Function input:y,y_hat
#Function output:the loss
#Function Action:calculate the loss
################################################
def loss_softmax(y,y_hat):
    min_val = 0.000000000001
    m = y.shape[0]
    #loss_ret = -1/m *(np.sum(y*np.log(y_hat.clip(min=min_val))))
    #loss_ret = np.sum(-y*np.log(y_hat))/m
    loss_ret=-1 * np.sum((y * np.log(y_hat.clip(min=min_val)))) / m
    return loss_ret

#################################################
#Function name:predict_y_hat
#Function input:model and train_X
#Function output: matrix predictions
#Function Action:the function create matrix predictions
################################################
def predict_y_hat(nn_model,x):
    c = forward_propagation_action(nn_model,x)
    y_hat = np.argmax(c['A2'],axis=1)
    return y_hat


#################################################
#Function name:find_accuracy
#Function input:y,y_hat
#Function output:accuracy
#Function Action:calculate accuracy
################################################
def find_accuracy(y,y_hat):
    #get total number of examples
    m=y.shape[0]
    #ensure prediction y_hat and truth vector y have the same size:
    y_hat = y_hat.reshape(y.shape)
    #calculate the number of wrong examples
    #error = np.sum(np.abs(y_hat-y))
    #calculate accuracy
    #total_loss = 100*(m-error)/m
    #return total_loss
    return np.sum(y==y_hat,axis=0)/float(m)


#################################################
#Function name:init_parameters
#Function input:dims
#Function output:initial parameters
#Function Action:Learning the w parameter allows us to
#  change the steepness (slope) of the line we are learning.
#  Learning a bias term as well, allows us to also shift the
#  function up or down and thus produce a better model for our data.
################################################
def init_parameters(n_input,n_hid,n_result):
    #first matrix of weights
    #w1 of size 784 x h=128
    w1 = np.random.uniform(low=-0.08, high=0.08, size=(n_input,n_hid))

    #first bias matrix 1x128
    b1 = np.random.uniform(low=-0.08, high=0.08, size=(1,n_hid))

    #second matrix of weights - 128x10
    w2 = np.random.uniform(low=-0.08, high=0.08, size=(n_hid,n_result))

    # second bias matrix 1x10
    b2 = np.random.uniform(low=-0.08, high=0.08, size=(1,n_result))
    #save the paramters as a model
    my_model = {'w1': w1,
                'b1':b1,
                'w2':w2,
                'b2':b2}
    return my_model


#################################################
#Function name:forward_propagation_action
#Function input:model and input matrix
#Function output:result model
#Function Action:This is the forward propagation function
#feed the neural networks firts time, and use activition function
#to convert values to be between 0 to 1
################################################
def forward_propagation_action(nn_model,A0):

    #first load the model parameters
    w1 = nn_model['w1']
    w2 = nn_model['w2']
    b1 = nn_model['b1']
    b2 = nn_model['b2']

    #compute Z1: input layer matrix dot w1 wheight matrix plus our bias
    #z1 - 44000 x 128
    z1 = np.dot(A0,w1)+b1

    #put it throgh our activition function
    A1=np.tanh(z1)

    # compute Z2: second linear step
    #w2 size - 128x10, A1 size = 44000x128, b2 = 10x1
    z2 = np.dot(A1,w2)+b2

    #now, we'll use the softmax as our activition function
    A2 = softmax(z2)#44000x10

    #save all results as a model
    result_model = {'A0':A0,
                    'z1':z1,
                    'A1':A1,
                    'z2':z2,
                    'A2':A2}
    return result_model


def loss_deriv(y,y_hat):
    return y_hat - y


#################################################
#Function name:back_propagation_action
#Function input:model network, model of forward propagation function results
#Function output:the gradients
#Function Action:his is the BACKWARD PROPAGATION function
#The backpropagation step involves the propagation of the neural network's
# error back through the network. Based on this error the neural network's
# weights can be updated so that they become better at minimizing the error.
################################################
def back_propagation_action(nn_model,nn_score,y):
    # first load the model parameters
    w1 = nn_model['w1']
    w2 = nn_model['w2']
    b1 = nn_model['b1']
    b2 = nn_model['b2']

    a0 = nn_score['A0']
    a1 = nn_score['A1']
    a2 = nn_score['A2']

    #get number of samples
    m = y.shape[0]

    #y_hat - y prediction - dz2
    delta2 = loss_deriv(y, a2)
   # delta1 = (delta2).dot(w2.T)*A1*(1-A1)


    #a1 - 44000x128 ->a1t 128x44000 , delta2 = 44000x10
    #dw2 = (a1.T).dot(delta2)#128x10
    dw2 = np.dot(a1.T,delta2)/m
   # dw2=dw2/m

    #1x10
    db2 = np.sum(delta2,axis=0)/m
    #db2 = (delta2).sum(axis=0)

    # delta2 - 44000x10, w2t - 10x128 - delta1- 44000x128
    delta1 = np.multiply(delta2.dot(w2.T), tanh_deriv_function(a1))

    #a0.T - 784x44000 delta1- 44000x128 -> dw1 - 784x128
    #dw1 = (1/m)*a0.T.dot(delta1)
    dw1 = np.dot(a0.T,delta1)/m

    #db1 = (1/m)*(delta1).sum(axis=0) - 1x128
    db1 = np.sum(delta1,axis=0)/m

    my_grades = {'dw2':dw2,
                 'db2':db2,
                 'dw1':dw1,
                 'db1':db1}
    return my_grades


#################################################
#Function name:update_my_params
#Function input:models and learning rate
#Function output
#Function Action:
################################################
def update_my_params(grades,learning_rate,nn_model):
    # Load parameters
    w1 = nn_model['w1']
    w2 = nn_model['w2']
    b1 = nn_model['b1']
    b2 = nn_model['b2']

    # Update parameters
    w1 -= grades['dw1']*learning_rate
    b1 -= grades['db1']*learning_rate
    w2 -= grades['dw2']*learning_rate
    b2 -= grades['db2']*learning_rate

    #save the paramters as a model after update rule
    nn_model = {'w1': w1,
                'b1':b1,
                'w2':w2,
                'b2':b2}
    return nn_model


#################################################
#Function name:final_test
#Function input:model, test and correct result
#Function output:none
#Function Action:compare between our model predictions and
#the correct tags
################################################
def final_test(final_model,test_x,test_y):
    feed_forward_model = forward_propagation_action(final_model, test_x)
    a2 = feed_forward_model['A2']
    print('Loss :', calc_loss(test_y, a2))
    y_hat = predict_y_hat(final_model, test_x)
    # Calculate accuracy
    print('Accuracy :', find_accuracy(test_y, y_hat) * 100, '%')


#################################################
#Function name:train_action
#Function input:models, epochs and our learning rate
#Function output:final model
#Function Action:train on the training set and then test the
#network on the test set. This has the network make predictions on data it has never seen
################################################
def train_action(nn_model,x,y,learning_rate,epochs):
    # Gradient descent. Loop over epochs
    for i in range(0,epochs):
        feed_forward_model = forward_propagation_action(nn_model, x)
        #print loss & accuracy every 100 iteration:
        grades = back_propagation_action(model, feed_forward_model, y)
        # call update_my_params function
        nn_model = update_my_params(grades, learning_rate, model)
        if i % 100 == 0:
            a2 = feed_forward_model['A2']
            print('Loss after iteration ', i, ':', calc_loss(y, a2))
            y_hat = predict_y_hat(nn_model, x)
            y_true = y.argmax(axis=1)
            print('Accuracy after iteration ', i, ':',find_accuracy(y_true,y_hat)*100,'%' )
    return nn_model

#################################################
#Function name:load_data
#Function input:none
#Function output:load needed data
#Function Action:load the data, sjuffel it and
#Split the train into train and validation with
# 80:20 ratio.  Use the validation set for hyper-parameters tuning
################################################

def load_data():
    train_x = np.loadtxt("train_x")
    #train_x = np.divide(train_x, 255.0)
    train_y = np.loadtxt("train_y")
    train_y = train_y.reshape(55000, 1)

    c = np.c_[train_x.reshape(len(train_x), -1), train_y.reshape(len(train_y), -1)]
    np.random.shuffle(c)
    train_x1 = c[:, :train_x.size // len(train_x)].reshape(train_x.shape)
    train_y1 = c[:, train_x.size // len(train_x):].reshape(train_y.shape)
    #train_x1 = np.divide(train_x1, 255.0)
    train_x1=train_x1/255

    dev_size = int(0.2 * train_x.shape[0])
    dev_x = train_x1[-dev_size:, :]
    dev_y = train_y1[-dev_size:]
    train_xx = train_x1[:-dev_size, :]
    train_yy = train_y1[:-dev_size]
    print (train_xx.shape[0])
    model_data = {'train_xx': train_xx,
                 'train_yy': train_yy,
                 'dev_x': dev_x,
                 'dev_y': dev_y}
    return model_data


if __name__ == '__main__':
    # netWork parameters:
    num_hidden_layer = 128  # 128 units in hidden hidden_layer
    num_input_layer = 784  # units in data input : image shape is 28x28
    num_output_layer = 10  # num of classes is 10 (0-9)

    model_data = load_data()
    np.random.seed(0)
    train_x, train_y = model_data['train_xx'], model_data['train_yy']
    dev_x, dev_y = model_data['dev_x'], model_data['dev_y']
    print(dev_x.shape)
    print(train_x.shape)
    print(train_x[0])
    model = init_parameters(num_input_layer, num_hidden_layer, num_output_layer)
    T = np.zeros((44000, 10))
    #create a matrix represent train_y
    for i in range(44000):
        T[i, int(train_y[i])] = 1
    print(T[0])
    print(train_y[0])
    model = train_action(nn_model=model, x=train_x, y=T, learning_rate=0.01, epochs=20000)
    final_test(model,dev_x,dev_y)
    test_x = np.loadtxt("test_x")

    #save reslut to test.pred file
    y_test = predict_y_hat(model, test_x)
    f = open("test.pred", "w")
    for x in y_test:
        f.write(str(x) + '\n')
    f.close()
    #save the result model params
    # w1 = model['w1']
    #
    # w2 = model['w2']
    # b1 = model['b1']
    # b2 = model['b2']
    # np.save("w1.bin", w1)
    # np.save("w2.bin", w2)
    # np.save("b1.bin", b1)
    # np.save("b2.bin", b2)
    #
    # np.save("dev_x.bin", dev_x)
    # np.save("dev_y.bin", dev_y)

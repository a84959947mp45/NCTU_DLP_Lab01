import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    return 1 / (1 + np.exp(-x))


def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return y * (1 - y)


class GenData:
    @staticmethod
    def _gen_linear(n=100):
        """ Data generation (Linear)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data = np.random.uniform(0, 1, (n, 2))

        inputs = []
        labels = []

        for point in data:
            inputs.append([point[0], point[1]])

            if point[0] > point[1]:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def _gen_xor(n=100):
        """ Data generation (XOR)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data_x = np.linspace(0, 1, n // 2)

        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def fetch_data(mode, n):
        """ Data gather interface

        Args:
            mode (str): 'Linear' or 'XOR', indicate which generator is used.
            n (int):    the number of data points generated in total.
        """
        assert mode == 'Linear' or mode == 'XOR'

        data_gen_func = {
            'Linear': GenData._gen_linear,
            'XOR': GenData._gen_xor
        }[mode]

        return data_gen_func(n)


class SimpleNet:
    def __init__(self, hidden_size,neural_node1,neural_node2, num_step=2000, print_interval=100):
        """ A hand-crafted implementation of simple network.

        Args:
            hidden_size:    the number of hidden neurons used in this model.
            num_step (optional):    the total number of training steps.
            print_interval (optional):  the number of steps between each reported number.
        """
        self.num_step = num_step
        self.print_interval = print_interval

        # Model parameters initialization
        # Please initiate your network parameters here.

        # the node of layer
        self.neural_node0 =2
        self.neural_node1 =neural_node1
        self.neural_node2 =neural_node2
        # weight
        self.hidden_weights =  np.random.randn(hidden_size+1,self.neural_node1,self.neural_node2)
        self.b = np.random.randn(hidden_size+1,self.neural_node1+1)
        #use compute
        self.z = np.zeros((hidden_size+1,self.neural_node1))
        self.a = np.zeros((hidden_size+2,self.neural_node1))
        #differential variable
        self.diff_z = np.zeros((hidden_size+1,self.neural_node1))
        self.diff_w = np.zeros((hidden_size+1,self.neural_node1,self.neural_node1))
        self.diff_b = np.zeros((hidden_size+1,self.neural_node1+1))
        #learning rate
        self.lr = 0.2

  
    @staticmethod
    def plot_result(data, gt_y, pred_y):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.

        Args:
            data:   the input data
            gt_y:   ground truth to the data
            pred_y: predicted results to the data
        """
        print(data.shape[0]," ",gt_y.shape[0])
        assert data.shape[0] == gt_y.shape[0]
        print(data.shape[0]," ",pred_y.shape[0])
        assert data.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if gt_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if pred_y[idx] == 0:

                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()
   
    #including input,hidden layer,output, we have four layers
    def forward(self, inputs):
        """ Implementation of the forward pass.
        It should accepts the inputs and passing them through the network and return results.
        """
        #initial
        self.z = np.zeros((3,self.neural_node1))
        self.a = np.zeros((4,self.neural_node1))

        #input layer ： input value input into a value
        self.a[0][0]=inputs[0][0]
        self.a[0][1]=inputs[0][1]

        #hidden layer1  
        '''
        z  =  a1 *w1 + a2 *w2 + b
        a  = sigmoid(z)
        '''
        for i in range(self.neural_node1) :
          self.z[0][i] =self.a[0][0]*self.hidden_weights[0][0][i]+self.a[0][1]*self.hidden_weights[0][1][i]+self.b[0][i]
          self.a[1][i]=sigmoid(self.z[0][i])
        
        #hidden layer2
        '''
        z  =  a1 *w1 + a2 *w2 + b
        a  = sigmoid(z)
        '''
        for i in range(self.neural_node2) :
            for j in range(self.neural_node1):
               self.z[1][i] += self.a[1][j]*self.hidden_weights[1][j][i]
            self.z[1][i] +=self.b[1][i]
            self.a[2][i]= sigmoid(self.z[1][i])

        #output layer
        '''
        z  =  a1 *w1 + a2 *w2 + b
        a  = sigmoid(z)
        '''
        for i in range(self.neural_node2) :
           self.z[2][0] += self.a[2][j]*self.hidden_weights[2][j][0]
        self.z[2][0] +=self.b[2][0]
        self.a[3][0]= sigmoid(self.z[2][0])
        
        return self.a[3][0]

    def backward(self):
        """ Implementation of the backward pass.
        It should utilize the saved loss to compute gradients and update the network all the way to the front.
        """
        #back output layer
        self.lr_w= np.zeros((3,self.neural_node1,self.neural_node1))
        self.lr_b= np.zeros((3,self.neural_node1))
        #diff_z = diff_loss * der_sigmoid(y)
        self.diff_z[2][0]=2*self.error*der_sigmoid(self.a[3][0])*self.a[3][0]
        #compute weight gradient : diff_w = a * diff_z
        for i in range(self.neural_node2):
            self.diff_w[2][i][0]=self.a[2][i]*self.diff_z[2][0]
        self.diff_b[2][0] = 1 * self.diff_z[2][0]
        

        #hidden layer2

        
        for i in range(self.neural_node2):
            #diff_z = diff_loss * der_sigmoid(y)
            self.diff_z[1][i] = der_sigmoid(self.a[2][i])*self.hidden_weights[2][i][0]*self.diff_z[2][0]

            for j in range(self.neural_node1):
                self.diff_w[1][j][i]=self.a[1][j]*self.diff_z[1][i]
            self.diff_b[1][i] = 1 * self.diff_z[1][i] 

        #hidden layer1
        for i in range(self.neural_node1):
            #diff_z = diff_loss * der_sigmoid(y)
            x1 = 0
            for z in range(self.neural_node2) :
               x1 += self.hidden_weights[1][i][z]*self.diff_z[1][z]
            self.diff_z[0][i] = der_sigmoid(self.a[1][i])*x1
            #compute weight gradient : diff_w = a * diff_z
            for j in range(self.neural_node0):
                self.diff_w[0][j][i]=self.a[0][j]*self.diff_z[0][i]
            self.diff_b[0][i] = 1 * self.diff_z[0][i] 
        

        #hidden layer weight  update
        for i in range(2):
            for j in range(self.neural_node1):
                for z in range(self.neural_node1):
                    self.hidden_weights[i][j][z]+=(-1)*(self.lr)*self.diff_w[i][j][z]
                self.b[i][j]+=(-1)*(self.lr)*self.diff_b[i][j]
                
        #output layer weight  update
        for i in range(self.neural_node1):
          self.hidden_weights[2][i][0]+=(-1)*(self.lr)*self.diff_w[2][i][0]
        self.b[2][0]+=(-1)*self.lr*self.diff_b[2][0]



    def train(self, inputs, labels):
        """ The training routine that runs and update the model.

        Args:
            inputs: the training (and testing) data used in the model.
            labels: the ground truth of correspond to input data.
        """
        # make sure that the amount of data and label is match
        assert inputs.shape[0] == labels.shape[0]

        # input size
        n = inputs.shape[0]
 
        all_loss=0

        xx=[]
        for epochs in range(self.num_step):
            all_loss=0
            for idx in range(n):
                # operation in each training step:
                #   1. forward passing
                #   2. compute loss
                #   3. propagate gradient backward to the front
              
                #preidct
                self.output = self.forward(inputs[idx:idx+1, :])
                #loss value
                self.error = self.output - labels[idx:idx+1, :]
                all_loss+=self.error**2
                self.backward()
            xx.append(all_loss)
      
            if epochs % self.print_interval == 0:
                print('Epochs {}: '.format(epochs))
                self.test(inputs, labels)

        x = np.arange(0,self.num_step)
        xx= np.array(xx)
        xx = np.squeeze(xx,1)
        xx = np.squeeze(xx,1)
        print(xx.shape)
        plt.plot(x,xx)
        plt.show()
        
        print('Training=finished')
        self.test(inputs, labels)

    def test(self, inputs, labels):
        """ The testing routine that run forward pass and report the accuracy.

        Args:
            inputs: the testing data. One or several data samples are both okay.
                The shape is expected to be [BatchSize, 2].
            labels: the ground truth correspond to the inputs.
        """
        n = inputs.shape[0]

        error = 0.0
        for idx in range(n):
            result = self.forward(inputs[idx:idx+1, :])
            error += abs(result - labels[idx:idx+1, :])

        error /= n
        print('accuracy: %.2f' % ((1 - error)*100) + '%')
        print('')


if __name__ == '__main__':

    #train data generator
    data, label = GenData.fetch_data('XOR', 70)  
    #create nn to train,it will train about 5000 times repeatly
    net = SimpleNet(2,4,4, num_step=5000)

    net.train(data, label)
    
    #produce all predict and transfer ndarray
    n = data.shape[0]
    #all predice array
    store = np.array([])
    for idx in range(n):
        output = net.forward(data[idx:idx+1, :])
        x = []
        x.append(output)
        store=np.append(store,np.array(x))
    
    #format predict data
    store = np.expand_dims(store,0)
    pred_result = np.round(store)
    #view answer
    SimpleNet.plot_result(data, label, pred_result.T)

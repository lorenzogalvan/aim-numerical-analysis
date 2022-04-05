import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

''' Istruzioni: devi inserire il tuo codice all'interno degli apici'''

class grid:
    def __init__(self, n):
        self.n_points = n
        self.x = tf.random.normal(shape=[self.n_points]).numpy()
        self.y = tf.random.normal(shape=[self.n_points]).numpy()
        self.xy = tf.stack((self.x,self.y), axis=1)


class NN:
    def __init__(self, u_ex, n_layers = 3,
                       n_neurons = 4,
                       activation = tf.nn.tanh,
                       dim = 2,
                       learning_rate = 1e-3,
                       opt = tf.keras.optimizers.Adam):

        '''
        Your code goes here

        Hints:

        self.hidden_layers =  ? Make a list of dense layers as n_hidden_layers with n_neurons each
        self.layers =  ?

        Keep this:
        '''

        '''
        self.hidden_layers = [tf.keras.layers.Dense(n_neurons, activation = activation) for i in range(n_layers - 1)]
        self.layers = [self.hidden_layers, tf.keras.layers.Dense(n_neurons, input_dim = dim, activation = activation)]
        self.n_neurons = n_neurons
        self.activation = activation
        self.model = tf.keras.Sequential(self.layers)'''

        # Here we first describe the general parameters of the NN

        self.n_layers = n_layers        # number of layers
        self.n_neurons = n_neurons      # number of neurons
        self.activation = activation    # activation function
        self.output_dim = 1             # output dimension
        self.input_dim = dim            # input dimension

        self.layers = [tf.keras.Input(shape = (self.input_dim,))] # First we create the input layer
        self.hidden_layers = [tf.keras.layers.Dense(n_neurons, activation = activation) for i in range(n_layers)]
        # Then we create all the other layers of our NN
        # We use n_layers (do not count input and output layer as hidden_layers)

        self.layers.extend(self.hidden_layers)  # We add all the layers to the input one
        self.layers.append(tf.keras.layers.Dense(self.output_dim, activation = activation)) # And we can finally append the output layer

        self.model = tf.keras.Sequential(self.layers)
        self.last_loss_fit = tf.constant([0.0])
        self.learning_rate = learning_rate
        self.optimizer = opt(learning_rate)
        self.u_ex = u_ex


    def __call__(self,val):
        return self.model(val)

    def __repr__(self):
        ''' Make a method to print the number of layers,
            neaurons, activation function, optimizer
            and learning rate of the NN'''

        # We create a string with all the needed information about our NN

        s = '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n' + \
            '                     NN PROPERTIES                                 \n' + \
            '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n' + \
            'Number of layers : ' + str(self.n_layers) + '\n' \
            'Number of neurons in each layer :  ' + str(self.n_neurons) + '\n' + \
            'Activation function :  ' + str(self.activation) + '\n' + \
            'Optimizer :  ' + str(self.optimizer) + '\n' + \
            'Learning rate: ' + str(self.learning_rate)
        return s

    def loss_fit(self,points):
        '''
        Using tf.reduce_mean and tf.square
        create the MSE for the interpolation loss

        pseudo code:
        MSE = 1/(nr points)*sum_(x,y \in points)(model(x,y)-u_ex(x,y))^2

        create the MSE for the interpolation loss
        create the MSE for the interpolation loss

        HINTS:
        self(points.xy) evaluate the NN in points
        self.u_ex(points.x,points.y) evaluate u_ex in points

        Be sure they have the same shape!

        self.last_loss_fit = ??
        '''

        # print(self(points.xy).shape)
        # print(self.u_ex(points.x, points.y))
        # They have the same shape (num_train_points, 1) - ok!

        self.last_loss_fit = tf.reduce_mean(tf.square(self(points.xy) - self.u_ex(points.x, points.y)))

        # self.last_loss_fit = MSE
        # We just created the MSE by using the suggested functions for the mean and the square

        return self.last_loss_fit

    def fit(self, points, log, num_epochs=100):
        '''
        Create una routine che minimizzi la loss fit
        e mostri il tempo impiegato
        '''
        # self.model.compile(optimizer = self.optimizer, loss = self.last_loss_fit)
        # self.loss_fit(points)
        epoch_time = time.time()    # We set a starting time
        epoch = 0                   # We initialize the epoch counter

        while epoch < num_epochs:   # We loop over the specified number of epochs
            epoch = epoch + 1
            to_min = lambda: self.loss_fit(points)  # We create the function to minimize, which is exaactly the MSE implemented before
            self.optimizer.minimize(to_min, self.model.variables)   # We can use the function minimize of the optimizer to advance in every epoch
            print("Time: ", time.time() - epoch_time, "epoch: ", epoch, "Loss: ", self.last_loss_fit.numpy())   # We show time, epoch and loss of the current iteration

        print("NN loss: ", self.last_loss_fit.numpy(), "\n", file=log)
        print("NN time: ", time.time() - epoch_time, "\n", file=log)    # We save this data into the log output file

        return

class PINN(NN):
    def __init__(self, u_ex, n_layers = 3,
                       n_neurons = 4,
                       activation = tf.nn.tanh,
                       dim = 2,
                       learning_rate = 1e-3,
                       opt = tf.keras.optimizers.Adam,
                       mu = tf.Variable(1.0),
                       inverse = False):
        super().__init__(u_ex, n_layers, n_neurons, activation, dim, learning_rate, opt)    # We can use super() to use the __init__ class of the NN

        '''
        Build father class
        '''

        self.mu = mu
        self.last_loss_PDE = tf.constant([0.0]);
        self.trainable_variables = [self.model.variables]
        if inverse:
            self.trainable_variables.append(self.mu)    # Now we can just add mu to the trainable variables thanks to the inheritance from father class

        '''
        Aggiungi self.mu alle trainable variables
        (oltre alle model.variables) quando
        vogliamo risolvere il problema inverso

        self.trainable_variables = ?
        '''


    def loss_PDE(self, points):
        '''
        Definite la lossPde del Laplaciano
        Guardate le slide per vedere come definire la PDE del Laplaciano

        Hints:
        x = tf.constant(points.x)
        y = tf.constant(points.y)
        with ...
            ...
            ...
            u = self.model(tf.stack((x,y),axis=1))
            u_x = ...
            u_y = ...
            u_xx = ...
            u_yy = ...
        self.last_loss_PDE = tf.reduce_mean(tf.square(-self.mu*(u_xx+u_yy)-tf.reshape(u,(x.shape[0],))))
        return self.last_loss_PDE
        '''
        x = tf.constant(points.x)
        y = tf.constant(points.y)

        with tf.GradientTape(persistent = True) as tape:
            tape.watch(x)
            tape.watch(y)
            u = self.model(tf.stack((x,y),axis=1))
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)

        self.last_loss_PDE = tf.reduce_mean(tf.square(-self.mu*(u_xx+u_yy)-tf.reshape(u,(x.shape[0],)))) # We create the loss for the PDE, again with reduce_mean and square

        return self.last_loss_PDE

    def fit(self,points_int,points_pde,log,num_epochs=100):
        '''
        Allena la rete usando sia la loss_fit che la loss_PDE
        '''
        time_fit = time.time()  # Starting time
        epoch = 0               # Epoch counter

        while epoch < num_epochs:
            epoch = epoch + 1
            to_min = lambda: self.loss_fit(points_int) + self.loss_PDE(points_pde)  # Now we try to minimize the sum of the MSE from the NN and from the PDE
            self.optimizer.minimize(to_min, self.trainable_variables)
            print("Time: ", time.time() - time_fit, "Epoch: ", epoch, "Loss NN: ",self.last_loss_fit.numpy(), "Loss PDE: ", self.last_loss_PDE.numpy(), "mu: ", self.mu.numpy())

        print("PINN: ", "\n", file=log)
        print("Elapsed time: ", time.time() - time_fit, "\n", file=log)
        print("NN loss: ", self.last_loss_fit.numpy(), "\n", file=log)
        print("PDE loss: ", self.last_loss_PDE.numpy(), "\n", file=log)
        # We save all the data in the log output file

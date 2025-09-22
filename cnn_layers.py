import numpy as np

def softmax(x): 
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

def cross_entropy(x): 
    return -np.log(x)

def regularized_cross_entropy(layers, lam, x): 
    loss = cross_entropy(x)
    for layer in layers: 
        loss += lam * (np.linalg.norm(layer.get_weights()) ** 2)
    return loss

def leaky_relu(x, alpha = 0.001): 
    return x * alpha if x < 0 else x

def leaky_relu_deriv(x, alpha = 0.001): 
    return alpha if x < 0 else 1

def lr_schedule(learning_rate, iteration): 
    if (iteration >= 0) and (iteration <= 10000): 
        return learning_rate
    if iteration > 10000: 
        return learning_rate * 0.1
    
class Convolutional: # convolution layer using 3x3 filters
    def __init__(self, name, num_filters = 16, stride = 1, size = 3, activation = None): 
        self.name = name
        self.filters = np.random.randn(num_filters, 3, 3) * 0.1
        self.stride = stride
        self.size = size
        self.activation = activation
        self.last_input = None
        self.leaky_relu = np.vectorize(leaky_relu)
        self.leaky_relu_deriv = np.vectorize(leaky_relu_deriv)
        
    
    def forward_prop(self, image): 
        self.last_input = image                                             # keeping track of last input for later backpropagation
        
        input_dim = image.shape[1]                                          # input dimension
        output_dim = int((input_dim - self.size)/self.stride) + 1           # output dimension
        
        out = np.zeros((self.filters.shape[0], output_dim, output_dim))     # creating matrix to hold values of the convolution
        
        for f in range(self.filters.shape[0]):                              # convolving each filter over the image
            tmp_y = out_y = 0                                               # first moving it vertically then horizontally
            while tmp_y + self.size <= input_dim: 
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dim: 
                    patch = image[: , tmp_y : tmp_y + self.size, tmp_x : tmp_x + self.size]
                    out[f, out_y, out_x] += np.sum(self.filters[f] * patch)
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
                
        if self.activation == 'relu': 
            self.leaky_relu(out)
            
        return out


    def back_prop(self, din, learning_rate = 0.005): 
        input_dim = self.last_input.shape[1]                                # input dimension

        if self.activation == 'relu':                                       # back propagate through relu
            self.leaky_relu_deriv(din)
        
        dout = np.zeros(self.last_input.shape)                              # loss gradient of the input through convolution operation
        dfilt = np.zeros(self.filters.shape)                                # loss gradient of filter
        
        for f in range(self.filters.shape[0]):                               # loop through all filters 
            tmp_y = out_y = 0
            while tmp_y + self.size <= input_dim: 
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dim:
                    patch = self.last_input[:, tmp_y : tmp_y + self.size, tmp_x : tmp_x + self.size]
                    dfilt[f] += np.sum(din[f, out_y, out_x] * patch, axis = 0)
                    dout[:, tmp_y : tmp_y + self.size, tmp_x : tmp_x + self.size] += din[f , out_y, out_x] * self.filters[f]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
                
        self.filters -= learning_rate * dfilt                               # update filters using SGD 
        
        return dout                                                         # return loss gradient for this layers inputs 

    
    def get_weights(self): 
        return np.reshape(self.filters, -1)
        

class Pooling:
    def __init__(self, name, stride = 2, size = 2): 
        self.name = name
        self.last_input = None
        self.stride = stride
        self.size = size
        
    def forward_prop(self, image): 
        self.last_input = image                                            # keeping track on last input for back prop
        
        num_channels, h_prev, w_prev = image.shape
        h = int((h_prev - self.size)/self.stride) + 1                       # computing dimensions after max pooling
        w = int((w_prev - self.size)/self.stride) + 1
        
        downsampled = np.zeros((num_channels, h, w))                        # holding values of max pooling
        
        for i in range(num_channels):                                       # sliding the window for every part of image
            curr_y = out_y = 0                                              # and taking max value
            while curr_y + self.size <= h_prev:                             # sliding window vertically across image
                curr_x = out_x = 0
                while curr_x + self.size <= w_prev:                         # sliding window horizontally across image
                    patch = image[i, curr_y : curr_y + self.size, curr_x : curr_x + self.size]
                    downsampled[i, out_y, out_x] = np.max(patch)            # choosing max value within the window at each step
                    curr_x += self.stride                                   # and store it in output matrix
                    out_x += 1
                curr_y += self.stride
                out_y += 1
        
        return downsampled
    
    
    def back_prop(self, din, learning_rate): 
        num_channels, orig_dim, *_ = self.last_input.shape                  # gradients are passed through the indices of 
                                                                            # greatest value in the original pooling during forward step
        
        
        dout = np.zeros(self.last_input.shape)                              # initializing the derivitive
        
        for c in range(num_channels): 
            tmp_y = out_y = 0
            while tmp_y + self.size <= orig_dim: 
                tmp_x = out_x = 0
                while tmp_x + self.size <= orig_dim: 
                    patch = self.last_input[c, tmp_y : tmp_y + self.size, tmp_x : tmp_x + self.size]        # obtain index of largest value in a patch
                    (x, y) = np.unravel_index(np.nanargmax(patch), patch.shape)
                    dout[c, out_y, out_x] += din[c, out_y, out_x]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
                
        return dout
    
    def get_weights(self): 
        return 0                                                            # pooling layers have no weights


class Fully_connected: 
    def __init__(self, name, nodes1, nodes2, activation): 
        self.name = name
        self.weights = np.random.randn(nodes1, nodes2) * 0.1
        self.biases = np.zeros(nodes2)
        self.activation = activation
        self.last_input_shape = None
        self.last_input = None
        self.last_output = None
        self.leaky_relu = np.vectorize(leaky_relu)
        self.leaky_relu_deriv = np.vectorize(leaky_relu_deriv)
        
    def forward_prop(self, input): 
        self.last_input_shape = input.shape                                 # keeping track of last input shape for back prop
        
        
        input = input.flatten()                                             # flattening input
        
        output = np.dot(input, self.weights) + self.biases                  # forward propagate
        
        if self.activation == 'relu':                                       # applying activation function 
            self.leaky_relu(output)
        
        self.last_input = input                                             # keeping track on last inputs and outputs
        self.last_output = output
        
        return output 
    
    
    def back_prop(self, din, learning_rate): 
        if self.activation == 'relu':                                       # back propagation through relu
            self.leaky_relu_deriv(din)
        
        self.last_input = np.expand_dims(self.last_input, axis = 1)
        din = np.expand_dims(din, axis = 1)
        
        dw = np.dot(self.last_input, np.transpose(din))                     # loss gradient of final dense layer weights
        db = np.sum(din, axis = 1).reshape(self.biases.shape)               # loss gradient of final dense layer biases
        
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
        
        dout = np.dot(self.weights, din)
        return dout.reshape(self.last_input.shape)
    
    def get_weights(self): 
        return np.reshape(self.weights, -1)
    

class Dense:                                                                # Dense layer with softmax activation
    def __init__(self, name, nodes, num_classes): 
        self.name = name
        self.weights = np.random.randn(nodes, num_classes) * 0.1
        self.biases = np.zeros(num_classes)
        self.last_input_shape = None
        self.last_input = None
        self.last_output = None
        
    def forward_prop(self, input): 
        self.last_input_shape = input.shape
        
        input = input.flatten()                                             # flatten input 
        
        output = np.dot(input, self.weights) + self.biases                  # forward propagate
        
        self.last_input = input
        self.last_output = output
        
        return softmax(output)
    
    def back_prop(self, din, learning_rate = 0.005):
        for i, gradient in enumerate(din):
            if gradient == 0:                                               # derivative of the loss function with respect to the output is non zero
                continue                                                    # only for the correct class so skip class if gradient is 0 
            
            t_exp = np.exp(self.last_output)                                # gradient of dout with respect to output
            dout_dt = -t_exp[i] * t_exp / (np.sum(t_exp) ** 2)
            dout_dt[i] = t_exp[i] * (np.sum(t_exp) - t_exp[i]) / (np.sum(t_exp) ** 2)
            
            dt = gradient * dout_dt                                         # gradient of loss with respect to output
            
            dout = self.weights @ dt                                        # gradient of loss with respect to input
            
            self.weights -= learning_rate * (np.transpose(self.last_input[np.newaxis]) @ dt[np.newaxis])
            self.biases -= learning_rate * dt
            
        return dout.reshape(self.last_input_shape)                          # return the loss gradient for this layers inputs
    
    def get_weights(self):
        return np.reshape(self.weights, -1)
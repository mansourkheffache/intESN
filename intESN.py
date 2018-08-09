import numpy as np
from copy import copy


# Shape of X
#   0:  number of entries
#   1:  time
#   2:  input signal dimensionality


class intESN:

    def __init__(self, N, K, L, q_input, q_output=lambda: 0, output_fb=False, clip=7):
        # quantization functions
        self.q_in = q_input
        self.q_out = q_output

        # init states and output weights
        self.states = np.zeros([N])
        self.W_out = np.zeros([K + N + 1, L])

        # store constants
        self.N = N      # reservoir size
        self.K = K      # input dims
        self.L = L      # output dims

        # clipping constant
        self.clip = clip

        # output feedback
        self.output_fb = output_fb
        self.last_output = np.zeros([L])

    def _init_optimizer(self, name, **kwargs):
        optimizers = {
            'sgd': {
                'params': ['lr'],
                'func': lambda w, dw, lr: w - (lr * dw),
            },
        }

        # check if chosen optimizer is supported
        if name not in optimizers.keys:
            raise Exception('Optimizer "{}" not supported. Bummer.'.format(name))

        # check for presence of all params
        for param in optimizers[name]['params']:
            if param not in kwargs:
                raise Exception('Parameter "{}" not provided to optimizer'.format(param))

        self.update_weights = optimizers[name]['func']

    def _init_loss_function(self, name, rr):
        # loss function anatomy:
        # 'name': lambda predicted, targets, n: -- return loss depending on those params
        loss_functions = {
            'cross_entropy': lambda predicted, targets, n: 0
        } 

        self.compute_data_loss = loss_functions[name]
        self.compute_reg_loss = lambda: 0.5 * rr * np.sum(self.W_out * self.W_out)

    def compile(self, task, loss, optimizer, rr=0.0, **kwargs):
        
        # check if task is supported
        supported_tasks = ['classification', 'regression']
        if task not in supported_tasks:
            raise Exception('Support for {} tasks not available yet'.format(self.task))
        self.task = task

        # init optimizer
        self._init_optimizer(optimizer, **kwargs)

        # define loss function
        self._init_loss_function(loss, rr)


    def _fix_input_dimensionality(self, X):
        # reshape data
        if X.ndim == 1:
            X = X[np.newaxis, :, np.newaxis]
        elif X.ndim == 2:
            X = X[np.newaxis, :]

        # sanity check
        if X.shape[2] != self.K:
            raise Exception('Input dimensionality mismatch: {} into {}'.format(X.shape[2], self.K))

    def _get_targets(self, y):
        # get proper targets
        targets = None
        if self.task == 'classification':
            if y.ndim != 1:
                raise Exception('Incorrect targets dimensionality')

            # sanity check
            if np.max(y) >= self.L:
                raise Exception('Number of classes beyond capacity: {} into {}'.format(np.max(y) + 1, self.L))

            targets = y
        elif self.task == 'regression':
            if y.ndim == 1:
                y = y[np.newaxis, :, np.newaxis]
            elif y.ndim == 2:
                y = y[np.newaxis, :]

            # sanity check
            if y.shape[2] != self.L:
                raise Exception('Output dimensionality mismatch: {} into {}'.format(y.shape[2], self.L))

            targets = y
            # get rid of last NaN, important for feedbacks as it becomes the first output feedback for t=0
            targets[:, -1] = np.zeros([L])
        else:
            raise Exception('Weird...')

        return targets

    def _reset_states(self):
        self.states = np.zeros([self.N])
        self.last_output = np.zeros([self.L])
        # first way is more efficient, but second more neat, find sweet spot
        # self.states *= not keep
        # self.last_output *= not keep

    def _harvest_states(self, X, targets, seq_length=2**32):

        extended_states = []

        # TODO use np.nan (NaN) to delimit signal
        # delimit all sequences with a NaN at the end of  - still TODO
        delimiter = np.argmax(X, axis=1)[:, 0]
        
        for i in X.shape[0]:

            self._reset_states()
            
            sequences = np.append(np.arange(0, delimiter, seq_length), delimiter)
            for interval in arange(sequences.size - 1):
                for t in arange(sequences[interval], sequences[interval + 1]):
                    # index: i * X.shape[1] + t
                    # use indexing to get proper targets too
                    # maybe consider output_fb in a different if case for efficiency? - PUT ME TO TEST!
                    self.states = self._clip(np.roll(self.states, 1) + self.q_in(X[i][t]) + self.output_fb * self.q_out(targets[i][t-1]))
                    # self.last_output = y[i][t]    # instead of nan_to_num ?
                extended_states.append(np.append(self.states, 1))

        extended_states = np.array(extended_states)

        return extended_states

    # def stimulate():
    def fit(self, X, y, discard=0, task='regression'):

        # reshape data
        if X.ndim == 1:
            X = X[np.newaxis, :, np.newaxis]
        elif X.ndim == 2:
            X = X[np.newaxis, :]

        # sanity check - input
        if X.shape[2] != self.K:
            print("incorrect input dimensionality")
            return -1

        # get proper targets
        if task == 'classification':
            if y.ndim != 1:
                print("incorrect targets dimensionality")
                return -1

            # fancy hot-one encoding
            targets = np.zeros((X.shape[0], self.L))
            targets[np.arange(X.shape[0]), y] = 1
            # targets[targets == 0] = -1        PENDEJO
        elif task == 'regression':
            if y.ndim == 1:
                y = y[np.newaxis, :, np.newaxis]
            elif y.ndim == 2:
                y = y[np.newaxis, :]

            # sanity check - output
            if y.shape[2] != self.L:
                print("incorrect targets dimensionality")
                return -1

            targets = y.flatten()       # might be useless
        else:
            print("invalid task type")
            return -1

        # harvest states
        extended_states = None
        if task == 'regression':

            # INFO making a list takes longer than going straight to numpy array. But if singal is interrupted (aka different lengths time-wise), then a lot of space will be wasted 

            # extended_states_list = []
            # TODO Use list instead for efficiency ??
            extended_states = np.zeros([X.shape[0] * X.shape[1], self.K + self.N + 1])

            for i in range(X.shape[0]):
                # reset states to 0
                self.states = np.zeros([self.N])

                for j in range(X.shape[1]):
                    self.states = self._clip(np.roll(self.states, 1) + self.q_in(X[i][j]) + self.output_fb * self.q_out(y[i][j-1]))
                    # extended_states_list.append(np.append(self.states, [X[i][j], 1]))
                    extended_states[i * X.shape[0] + j] = np.append(self.states, [X[i][j], 1])

            # filter out states
            # extended_states = np.array(extended_states_list, dtype='float')

            # discard transient states
            transient = int(extended_states.shape[0] * discard)
            extended_states = extended_states[transient:]
            targets = targets[transient:]
        elif task == 'classification':
            extended_states = np.zeros([X.shape[0], self.N + 1])
            for i in range(X.shape[0]):
                # reset states to 0
                self.states = np.zeros([self.N])

                for j in range(X.shape[1]):
                    self.states = self._clip(np.roll(self.states, 1) + self.q_in(X[i][j]) + self.output_fb * self.q_out(y[i]))
                    # print(self.states)
                    # extended_states[i * X.shape[0] + j] = np.append(self.states, [X[i][j], 1])

                extended_states[i] = np.append(self.states, [1])

        # compute weights
        self.W_out = np.dot(np.linalg.pinv(extended_states), targets)

        # print(extended_states)
        # print(self.W_out)
        # print(targets)

        # get RMSE
        pred = np.dot(extended_states, self.W_out)
        rmse = np.sqrt(np.mean((pred - y)**2))
        print(rmse)

        # store last output for future
        # self.last_output = y[-1][-1]

    def predict(self, X, y=None, reset=True):

        # reshape data
        if X.ndim == 1:
            X = X[np.newaxis, :, np.newaxis]
            # y = y[np.newaxis, :, np.newaxis]
        elif X.ndim == 2:
            X = X[np.newaxis, :]
            # y = y[np.newaxis, :]
 
        # sanity checks
        if X.shape[2] != self.K:
            print("incorrect input dimensionality")
            return

        pred = np.zeros([X.shape[0], self.L])
        if reset:
                self.states = np.zeros(self.N)
        # else:
        #     pred[0][-1] = self.last_output


        for i in range(X.shape[0]):
            self.states = np.zeros([self.N])

            for j in range(X.shape[1]):
                self.states = self._clip(np.roll(self.states, 1) + self.q_in(X[i][j]) + self.output_fb * 1) #self.q_out(pred[i][j-1]))
                # pred[i][j] = np.dot(np.append(self.states, [X[i][j], 1]), self.W_out)

            pred[i] = np.dot(self.W_out.T, np.append(self.states, [1]))

        if y is not None:
            rmse = np.sqrt(np.mean((pred - y)**2))
            print(rmse)

        return pred

    def softmax(self, X, y):

        # fancy hot-one encoding
        # targets = np.zeros((X.shape[0], self.L))
        # targets[np.arange(X.shape[0]), y] = 1



        # harvest states
        extended_states = np.zeros([X.shape[0], self.N + 1])
        for i in range(X.shape[0]):
            # reset states to 0
            self.states = np.zeros([self.N])
            for j in range(X.shape[1]):
                self.states = self._clip(np.roll(self.states, 1) + self.q_in(X[i][j]))

            extended_states[i] = np.append(self.states, [1])

        # random weights
        self.W_out = 0.01 * np.random.randn(self.N + 1, self.L)

        step_size = 1e-2
        reg = 1e-3

        # e stands for epoch
        for e in range(1500):
            # get scores and avoid exp-losion
            scores = np.dot(extended_states, self.W_out)
            scores -= np.max(scores)

            # compute the class probabilities
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

            # compute the loss: average cross-entropy loss and regularization
            # after a "batch"
            correct_logprobs = -np.log(probs[range(X.shape[0]), y])
            data_loss = np.sum(correct_logprobs) / X.shape[0]
            reg_loss = 0.5 * reg * np.sum(self.W_out * self.W_out)
            loss = data_loss + reg_loss

            if e % 10 == 0:
                print(loss)

            # compute the gradient on scores
            dscores = probs
            dscores[range(X.shape[0]), y] -= 1
            dscores /= X.shape[0]

            # backpropate the gradient to the parameters (W,b)
            dW = np.dot(extended_states.T, dscores)
            # db = np.sum(dscores, axis=0, keepdims=True)

            dW += reg * self.W_out # regularization gradient

            # perform a parameter update
            self.W_out += -step_size * dW

    def classify(self, X, y=None):

        # get predicted classes
        scores = self.predict(X)
        guesses = np.argmax(scores, axis=1)

        # get accuracy
        if y is not None:
            hits = guesses[guesses == y].size
            accuracy = float(hits) / y.size
            print(accuracy) 

        return guesses

    def _clip(self, b):
        """
        Clipping function, bounds the values of the activations
        """
        a = copy(b)
        a[a > self.clip] = self.clip
        a[a < -self.clip] = -self.clip
        return a

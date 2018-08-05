import numpy as np
from copy import copy


# Shape of X
#   0:  number of entries
#   1:  time
#   2:  dimensionality

class intESN:

    def __init__(self, N, K, L, q_input, q_output = lambda x: 0, output_fb = False, clip = 7):
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


    def fit(self, X, y, discard=0):

        # reshape data
        if X.ndim == 1:
            X = X[np.newaxis, :, np.newaxis]
            y = y[np.newaxis, :, np.newaxis]
        elif X.ndim == 2:
            X = X[np.newaxis, :]
            y = y[np.newaxis, :]

        # sanity checks
        if X.shape[2] != self.K:
            print("incorrect input dimensionality")


        # harvest extended states

        # making a list takes longer than going straight to numpy array. But if singal is interrupted (aka different lengths time-wise), then a lot of space will be wasted 

        extended_states_list = []
        # TODO Use list instead for efficiency ??
        # extended_states = np.zeros([X.shape[0] * X.shape[1], self.K + self.N + 1])

        for i in range(X.shape[0]):
            # reset states to 0
            self.states = np.zeros([self.N])

            for j in range(X.shape[1]):
                self.states = self._clip(np.roll(self.states, 1) + self.q_in(X[i][j]) + self.output_fb * self.q_out(y[i][j-1]))
                extended_states_list.append(np.append(self.states, [X[i][j], 1]))
                # extended_states[i * X.shape[0] + j] = np.append(self.states, [X[i][j], 1])

        # flat and smooth
        targets = y.flatten()       # might be useless
        extended_states = np.array(extended_states_list, dtype='float')

        # disregard transient states
        # print(discard)
        # print(extended_states.shape)
        # print(int(extended_states.size * discard))
        extended_states = extended_states[int(extended_states.shape[0] * discard):]
        targets = targets[int(targets.shape[0] * discard):]
        # print(extended_states.size)

        # compute weights
        self.W_out = np.dot(np.linalg.pinv(extended_states), targets)

        # get RMSE
        pred = np.dot(extended_states, self.W_out)
        rmse = np.sqrt(np.mean((pred - y)**2))
        print(rmse)

        # store last output for future
        self.last_output = y[-1][-1]

    def predict(self, X, y=None, reset=True):

        # reshape data
        if X.ndim == 1:
            X = X[np.newaxis, :, np.newaxis]
            y = y[np.newaxis, :, np.newaxis]
        elif X.ndim == 2:
            X = X[np.newaxis, :]
            y = y[np.newaxis, :]
 
        pred = np.zeros(y.shape)
        if reset:
            self.states = np.zeros(self.N)
        else:
            pred[0][-1] = self.last_output

        print(pred.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                self.states = self._clip(np.roll(self.states, 1) + self.q_in(X[i][j]) + self.output_fb * self.q_out(pred[i][j-1]))
                pred[i][j] = np.dot(np.append(self.states, [X[i][j], 1]), self.W_out)

        if y is not None:
            rmse = np.sqrt(np.mean((pred - y)**2))
            print(rmse)

        return pred


    def _clip(self, b):
        """
        Clipping function, bounds the values of the activations
        """
        a = copy(b)
        a[a > self.clip] = self.clip
        a[a < -self.clip] = -self.clip
        return a

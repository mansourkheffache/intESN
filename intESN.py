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


    def fit(self, X, y):
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

        # compute weights
        self.W_out = np.dot(np.linalg.pinv(extended_states), targets)

        # get RMSE
        pred = np.dot(extended_states, self.W_out)
        rmse = np.sqrt(np.mean((pred - y)**2))
        print(rmse)


    def predict(self, X, y=None, reset=True, last_output=0):

        pred = np.zeros(X.shape)

        if reset:
            self.states = np.zeros(self.N)
        else:
            pred[:][-1] = last_output

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

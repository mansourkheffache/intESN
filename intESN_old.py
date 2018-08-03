import numpy as np
from copy import copy

class intESN:

    def __init__(self, K, N, L, input_quantization, output_quantization=lambda x: 0, output_feedback=False, clipping=7, verbose=False, loss='mse', task='regression'):
        self.q_in = input_quantization
        self.q_out = output_quantization
        self.output_feedback = output_feedback
        self.clipping = clipping

        self.states = np.zeros([N])
        self.extended_states = np.zeros([K + N + 1])
        self.W_out = np.zeros([K + N + 1, L])

        self.K = K
        self.N = N
        self.L = L

        self.verbose = verbose

        self.loss = loss
        self.task = task

    def fit(self, X, y, epochs=1, optimizer='pinv', batch_size=32, lr=0.01, rr=0.001, discard=0.1):

        pred = None

        for epoch in range(1, epochs + 1):
            self.extended_states = np.zeros([X.shape[0] * X.shape[1], self.states.shape[0] + X.shape[0] + 1])

            for i in range(X.shape[0]):
                self.states = np.zeros(self.states.shape)

                for j in range(X.shape[1]):
                    self.states = self._clip(np.roll(self.states, 1) + self.q_in(X[i][j]) + self.output_feedback * self.q_out(y[i][j-1]))
                    self.extended_states[i * X.shape[0] + j] = np.append(self.states, [X[i][j], 1])

            transient = int(discard * self.extended_states.shape[0])
            self.W_out = np.dot(np.linalg.pinv(self.extended_states[transient:]), y[i][transient:])
            pred = np.reshape(np.dot(self.extended_states, self.W_out), (1, -1))

        error = np.sqrt(np.mean((pred - y)**2))
        print(error)


    def predict(self, X, y=None, reset=True):
        if reset:
            self.states = np.zeros(self.states.shape)

        # if X.ndim < 2:
        #     X = np.reshape(X, (1, -1))
        #     if y != None:
        #         y = np.reshape(y, (1, -1))

        # pred = np.zeros([X.shape[1]])
        pred = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                self.states = self._clip(np.roll(self.states, 1) + self.q_in(X[i][j]) + self.output_feedback * self.q_out(pred[i][j-1]))
                pred[i][j] = np.dot(np.append(self.states, [X[i][j], 1]), self.W_out)

        if y is not None:
            error = np.sqrt(np.mean((pred - y)**2))
            print(error)

        return pred

        # if y.shape[0] == 1:
        #     y = y.flatten

    def _clip(self, b):
        """
        Clipping function, bounds the values of the activations
        """
        a = copy(b)
        a[a > self.clipping] = self.clipping
        a[a < -self.clipping] = -self.clipping
        return a

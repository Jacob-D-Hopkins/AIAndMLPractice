import numpy as np

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

def forward(x):
    return w * x

def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y)/x.size

print(f'Prediction before training: f(5) = {forward(5):.3f}')

learning_rate = .01
n_iters = 20

for epoch in range(n_iters):
    #prediction
    y_predicted = forward(X)

    l = loss(Y, y_predicted)

    dw = gradient(X, Y, y_predicted)

    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
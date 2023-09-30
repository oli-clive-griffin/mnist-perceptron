import numpy as np

INPUT_SIZE = 28 * 28
LAYERS = [256, 128, 10]

class LinearLayer:
  def __init__(self, i, o):
    self.W = (np.random.rand(i, o) - 0.5) * 0.3 # this can be way better
    self.b = (np.random.rand(o) - 0.5) * 0.3

  def __call__(self, x):
    self.input = x
    return (x @ self.W) + self.b

  def back_wrt_W(self, upstream):
    return np.array([self.input]).T @ np.array([upstream])

  def back_wrt_b(self, upstream):
    return upstream

  def back_wrt_input(self, upstream):
    return upstream @ self.W.T
    

class ReLU:
  def __call__(self, x):
    self.input = x
    return np.maximum(0, x)

  def back_wrt_input(self):
    out = (self.input >= 0).astype(int)
    return out

class Softmax:
  def __call__(self, x):
    self.output = np.exp(x) / np.sum(np.exp(x))
    return self.output

class MLP:
  def __init__(self):
    self.l1 = LinearLayer(INPUT_SIZE, LAYERS[0])
    self.r1 = ReLU()
    self.l2 = LinearLayer(LAYERS[0], LAYERS[1])
    self.r2 = ReLU()
    self.l3 = LinearLayer(LAYERS[1], LAYERS[2])
    self.r3 = ReLU()
    self.sm = Softmax()

  def forward(self, x):
    z1 = self.l1(x)
    a1 = self.r1(z1)

    z2 = self.l2(a1)
    a2 = self.r2(z2)

    z3 = self.l3(a2)
    a3 = self.r3(z3)

    pred = self.sm(a3)

    return pred

  def backward(self, y, y_pred, learning_rate):
    grad = y_pred - y

    print(y)
    print(y_pred)
    print(grad)
    print('')

    grad = self.r3.back_wrt_input()
    grad_W3 = self.l3.back_wrt_W(grad)
    grad_b3 = self.l3.back_wrt_b(grad)
    grad = self.l3.back_wrt_input(grad)

    grad = self.r2.back_wrt_input()
    grad_W2 = self.l2.back_wrt_W(grad)
    grad_b2 = self.l2.back_wrt_b(grad)
    grad = self.l2.back_wrt_input(grad)

    grad = self.r1.back_wrt_input()
    grad_W1 = self.l1.back_wrt_W(grad)
    grad_b1 = self.l1.back_wrt_b(grad)
    grad = self.l1.back_wrt_input(grad)

    self.l3.W += (learning_rate * grad_W3)
    self.l3.b += (learning_rate * grad_b3)
    self.l2.W += (learning_rate * grad_W2)
    self.l2.b += (learning_rate * grad_b2)
    self.l1.W += (learning_rate * grad_W1)
    self.l1.b += (learning_rate * grad_b1)

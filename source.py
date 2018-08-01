import numpy as np
import time
import numpy.random as rnd
import numpy.linalg as la
import pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lin


# Question 1

# 1 a)

def mymult(A, B):

  I, K = np.shape(A)
  K, J = np.shape(B)
  C = np.zeros([I, J])

  for i in range(I):

    for j in range(J):
      x = 0.0

      for k in range(K):
        x = x + A[i, k] * B[k, j]

      C[i, j] = x

  return C
        
    
# 1 b)

def mymeasure(I, K, J):

  A = np.random.rand(I, K)
  B = np.random.rand(K, J)
  t1 = time.time()
  C1 = np.matmul(A, B)
  t2 = time.time()
  print('')
  print('Execution time for numpy.matmul ({}, {}, {}: {}'.format(I, J, K, t2 - t1))
  t1 = time.time()
  C2 = mymult(A, B)
  t2 = time.time()
  print('Execution time for mymult ({}, {}, {}): {}'.format(I, J, K, t2 - t1))
  mag = np.sum((C1 - C2) ** 2)
  print('Magnitude of C1 - C2: {}'.format(mag))

# 1 c)

print('\n')
print('QUESTION 1(c).')
print('--------------')
mymeasure(1000, 50, 100)
mymeasure(1000, 1000, 1000)



# Question 3

with open('data1.pickle','rb') as f:
    dataTrain, dataTest = pickle.load(f)
    
Xtrain = dataTrain[:, 0]
Ytrain = dataTrain[:, 1]
Xtest = dataTest[:, 0]
YTest = dataTest[:, 1]
Ntrain = len(Ytrain)
Ntest = len(YTest)
yMax = 15.0
yMin = -15.0
xMin = 0.0
xMax = 1.0

 
# 3 a)

def dataMatrix(X, M):

  x = np.reshape(x, [-1])
  N = len(x)
  Z = np.ones([N, M + 1])

  for n in range(1, M + 1):
    Z[:, n] = Z[:, n - 1] * x

  return Z

def error(Y, Z, w):

  return np.mean((Y - np.matmul(Z, w)) ** 2)


def fitPoly(M):

  Ztrain = dataMatrix(Xtrain, M)
  w = la.lstsq(Ztrain, Ytrain)[0]
  errTrain = error(Ytrain, Ztrain, w)
  Ztest = dataMatrix(Xtest, M)
  errTest = error(Ytest, Ztest, w)
  eturn (w, errTrain, errTest)
    
    
# 3 c)


def plotPoly(w):

  w = np.reshape(w, [-1])
  M = len(w) - 1
  x = np.linspace(xMin, xMax, 1000)
  Z = dataMatrix(x, M)
  y = np.matmul(Z, w)
  plt.plot(x, y, 'r')
  plt.plot(Xtrain, Ytrain, 'b.')
  plt.ylim(yMin, yMax)

    
# 3 d)
def bestPoly():

  errTrainList = []
  errTestList = []
  errTestMin = np.Inf
  plt.figure()
  plt.suptitle('Question 3(d): best-fitting polynomials of degree M = 0, 1, .... , 15')

  for m in range(15 + 1):
    w, errTrain, errTest = fitPoly(m)
    errTrainList.append(errTrain)
    errTestList.append(errTest)
    plt.subplot(4, 4, m + 1)
    plotPoly(w)

    if errTest < errTestMin:
      errTestMin = errTest
      M = m
      w_best = w

  plt.figure()
  plt.plot(errTrainList, 'b')
  plt.plot(errTestList, 'r')
  plt.ylim(0, 250)
  plt.title('Question 3: training and test error')
  plt.xlabel('polynomial degree')
  plt.ylabel('error of best-fitting polynomials')
  plt.show()
  plt.figure()
  plotPoly(w_best)
  plt.title('Question 3: best-fitting polynomials (degree = {})'.format(M))
  plt.xlabel('x')
  plt.ylabel('y')
  print('Degree of best-fitting polynomials: {}'.format(M))
  print('')
  print('Weight vector of best-fitting polynomials: ')
  print(w_best)
  print('')
  print('Training and test error of best-fitting polynomials: ')
  print('   {},   {}'.format(errTestList[M], errTestList[M]))

print('\n')
print(' QUESTION 3(d). ')
print('--------------')
print('')
bestPoly()

        

# Question 4    


with open('data2.pickle','rb') as f:
  dataVal, dataTest = pickle.load(f)

Xval = dataVal[:, 0]
Yval = dataVal[:, 1]
Xtest = dataTest[:, 0]
Ytest = dataTest[:, 1]
Nval = len(Yval)
Ntest = len(YTest)

 
# 4 a)       
def fitRegPoly(M,alpha):

  Ztrain = dataMatrix(Xtrain, M)
  ridge = lin.Ridge(alpha)
  ridge.fit(Ztrain, Ytrain)
  w = ridge.coef_
  w[0] = ridge.intercept_
  errTrain = error(Ytrain, Ztrain, w)
  Zval = dataMatrix(Xval, M)
  errVal = error(Yval, Zval, w)
  return w, errTrain, errVal


# 4 b)
def bestRegPoly():

  M = 15
  errTestList = []
  errValList = []
  errValMin = np.Inf
  alphaList = 10.0 ** np.arrage(-13, 3)
  plt.figure()
  plt.suptitle('Question 4(b): best-fitting polynomials for log(alpha) = -13, -12, ... , 1, 2')

  for i in range(15 + 1):
    alpha = alphaList[i]
    w, errTrain, errVal = fitRegPoly(M, alpha)
    errTrainList.append(errTrain)
    errValList.append(errVal)
    plt.subplot(4, 4, i+1)
    plotPoly(w)

    if errVal < errValMin:
      errValMin = errVal
      I = i
      w_best = w

  plt.figure()
  plt.semilogx(alphaList, errTrainList, 'b')
  plt.semilogx(alphaList, errValList, 'r')
  plt.title('Question 4: best-fitting polynomials (alpha = {})'.format(alphaList(I)))
  plt.xlabel('x')
  plt.ylabel('y')
  print('')
  print('Optimal value of alpha: {}'.format(alphaList(I)))
  print('')
  print('Weight vector of best-fitting polynomials: ')
  print(w_best)
  Ztest = dataMatrix(Xtest, M)
  errTest = error(Ytest, Ztest, w_best)
  print('')
  print('Training validation and test errors of best-fitting polynomials: ')
  print('   {},   {}    {}'.format(errTrainList[I], errValMin, errTest))


print('\n')
print('QUESTION 4(b)')
print('--------------')
bestRegPoly()



# Question 5
# 5  c)

def linGrad(Z, t, w):
  t = np.reshape(t, [-1, 1])
  w = np.reshape(w, [-1, 1])
  yhat = np.matmul(Z, w)
  err = t - yhat
  err = np.reshape(err, [1, -1])
  grad = -2.0 * np.matmul(err, Z)
  return np.reshape(grad, [-1])

    
def regGrad(Z, t, w, alpha):
  grad = 2.0 * alpha * w
  grad[0] = 0.0
  grad += linGrad(Z, t, w)
  return grad
    
# 5 d)
    
def fitPolyGrad(m,alpha,lrate):
  w = rnd.randn(M + 1)
  Ztrain = dataMatrix(Xtrain, M)
  errTrainList = []
  errTestList = []
  iList = 10 ** np.array([0, 1, 2, 3, 4, 5, 6, 7])
  j = 0
  plt.figure()
  plt.suptitle('Question 5(d): fitted polynomial as number of weight-updates increases')

  for i in range(10000000 + 1):
    grad = regGrad(Ztrain, Ytrain, w, alpha)
    w -= lrate * grad

    if np.mod(i, 1000) == 0:
      yhat = np.matmul(Ztrain, w)
      errTrain = np.sum((Ytrain - yhat) ** 2)/Ntrain
      errTrainList.append(errTrain)
      yhat = np.matmul(Ztest, w)
      errTest = np.sum((Ytest - yhat) ** 2)/Ntest
      errTestList.append(errTest)

    if i == iList[j]:
      j += 1
      plt.subplot(3, 3, j)
      plotPoly(w)

    if np.mod(i, 100000):
      print('iteration{},   Training error = {}'.format(i, errTrain))

  plt.figure()
  plotPoly(w)
  plt.title('Question 5: fitted polynomial')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.figure()
  plt.plot(errTrainList, 'b')
  plt.plt(errTestList, 'r')
  plt.title('Question 5: training and test error v.s. time')
  plt.xlabel('Number of iterations (in thousands)')
  plt.ylabel('Error')
  print('')
  print('Final training and test errors: ')
  print('   {},   {}'.format(errTrain, errTest))
  print('')
  print('Final weight vector: ')
  print(w)


print('\n')
print(' QUESTION 5 (d)')
print('--------------')
fitPolyGrad(15, 10.0 ** (-5), 0.01)
print('')





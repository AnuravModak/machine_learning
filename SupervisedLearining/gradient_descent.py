import numpy as np

def gradient_descent(x,y):
    m_curr,b_curr=0,0
    iterations=10000
    # iterations is also guessed
    n=len(x)
    learning_rate =0.08
    # learning rate is predicted by hit and trial.....

    for i in range(iterations):

        y_predicted=m_curr*x +b_curr
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        md=-(2/n)*sum(x*(y-y_predicted))
        bd=-(2/n)*sum(y-y_predicted)
        m_curr=m_curr-learning_rate*md
        b_curr=b_curr-learning_rate*bd
        print("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost,i))




x=np.array([1,2,3,4,5])
# print(x)
y=np.array([5,7,9,11,13])
gradient_descent(x,y)
# now in the output the "m" and "b" are reaching to their expected optimal values....
# i.e 2 and 3 and cost is also decreasing

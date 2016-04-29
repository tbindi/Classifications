from sklearn.linear_model import LogisticRegression
from LogisticRegression import MultiLogistic
from InputData import get_plain, get_test
from numpy import array


def test_LR():
    data, target = get_plain('../kddcup.data_10_percent.out')
    test_data, test_target = get_test('../test.data.out')
    classify = LogisticRegression(max_iter=100, multi_class='ovr')
    classify.fit(X=data, y=target)
    for i in array(test_data):
        print classify.predict(i)


def test_multi():
    x = array([[1, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0]])
    y = array([[1, 0, 0],
                 [1, 0, 0],
                 [1, 0, 0],
                 [0, 1, 0],
                 [0, 1, 0],
                 [0, 0, 1],
                 [0, 0, 1]])
    # data, target, features, classes
    classify = MultiLogistic(inp=x, clas=y, m=6, n=3)
    lr = 0.1
    cost_list = list()
    for i in xrange(300):
        classify.learn(lr=lr)
        cost = classify.neg_log_like()
        cost_list.append(cost)
        lr *= 0.95
    test = array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]])
    print classify.predict(test[0])


def test_logis():
    x = array([[1, 1, 1, 0, 0, 0],
               [1, 0, 1, 0, 0, 0],
               [1, 1, 1, 0, 0, 0],
               [0, 0, 1, 1, 1, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 1, 0],
               [0, 0, 1, 1, 1, 0]])
    y = array([0, 0, 1, 1, 2, 2, 2])
    classify = LogisticRegression(max_iter=300, multi_class='ovr')
    classify.fit(X=x, y=y)
    test = array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]])
    print classify.predict_proba(test[1])


def test_MLR():
    data, target = get_plain('../kddcup.data_10_percent.out')
    test_data, test_target = get_test('../test.data.out')
    classify = MultiLogistic(inp=data, clas=target, m=len(data[0]), n=5)
    lr = 0.1
    cost_list = list()
    for i in xrange(100):
        classify.learn(lr=lr)
        cost_list.append(classify.neg_log_like())
        lr *= 0.95
    for i in test_data:
        print classify.predict(i)


if __name__ == "__main__":
    # test_LR()
    test_MLR()
    # test_multi()
    # test_logis()



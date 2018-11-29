from sklearn import metrics


def make_hyphotesis_function(theta):
    def fn(size_in_meters):
        return float(theta[0]) + float(theta[1]) * float(size_in_meters)
    return fn


def cost_function(theta, training_set):
    sum_of_squared_errors = 0.0
    hyphotesis_function = make_hyphotesis_function(theta)
    for example in training_set:
        sum_of_squared_errors += (hyphotesis_function(example[0]) - example[1]) ** 2
    return (1.0 / 2.0 * len(training_set)) * sum_of_squared_errors


def gradient_descent_1(training_set, learning_rate=0.0009, interactions=1000):
    theta = (0.0, 0.0)
    m = float(len(training_set))
    for i in range(0, interactions):
        hyphotesis_function = make_hyphotesis_function(theta)
        theta_0_sum = sum([hyphotesis_function(x) - y for (x, y) in training_set])
        theta_1_sum = sum([(hyphotesis_function(x) - y) * x for (x, y) in training_set])
        theta = (
            theta[0] - (learning_rate / m) * theta_0_sum,
            theta[1] - (learning_rate / m) * theta_1_sum,
        )
    return theta


def house_price(size_in_meters):
    return 1.0 + 1.5 * size_in_meters


if __name__ == '__main__':
    dataset = [(float(x), float(house_price(x))) for x in range(10, 100, 10)]

    training_set = dataset[0:-2]
    testing_set = dataset[-2:]

    print('Traning set is {}'.format(training_set))
    print('Testing set is {}'.format(testing_set))
    print('\nRunning gradient descent:\n')

    theta = gradient_descent_1(training_set)
    predictions = [
        make_hyphotesis_function(theta)(size) for (size, price) in testing_set
    ]
    targets = [price for (size, price) in testing_set]
    mean_squared_error = metrics.mean_squared_error(predictions, targets)

    print('\tresult => {}'.format(theta))
    print('\tlabels => {}'.format(targets))
    print('\tpredictions => {}'.format(predictions))
    print('\tmean_squared_error => {}\n'.format(mean_squared_error))

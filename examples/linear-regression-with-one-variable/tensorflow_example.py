import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.python.data import Dataset

feature_columns = [tf.feature_column.numeric_column('size_in_meters')]

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.009)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=optimizer
)


def input_function(features, targets, batch_size=10, shuffle=True, num_epochs=None):
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    # TODO: Adjust it based on https://www.tensorflow.org/guide/datasets

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels


def house_price(size_in_meters):
    return 1.0 + 1.5 * size_in_meters


if __name__ == '__main__':
    dataset = np.array([[float(x), float(house_price(x))] for x in range(10, 100, 10)])
    features = {'size_in_meters': dataset[:, 0]}
    targets = dataset[:, 1]

    linear_regressor.train(
        input_fn=lambda: input_function(features, targets),
        steps=100
    )

    predictions = linear_regressor.predict(
        input_fn=lambda: input_function(features, targets, num_epochs=1, shuffle=False)
    )
    predictions = np.array([item['predictions'][0] for item in predictions])

    mean_squared_error = metrics.mean_squared_error(predictions, targets)
    weight = linear_regressor.get_variable_value('linear/linear_model/size_in_meters/weights')[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
    theta = [bias.tolist()[0], weight.tolist()[0]]

    print('\tresult => {}'.format(theta))
    print('\tlabels => {}'.format(targets.tolist()))
    print('\tpredictions => {}'.format(predictions.tolist()))
    print('\tcost => {}\n'.format(mean_squared_error))

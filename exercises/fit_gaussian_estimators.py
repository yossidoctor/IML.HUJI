import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    samples = np.random.normal(mu, sigma, size=1000)
    model = UnivariateGaussian().fit(samples)
    print((np.round(model.mu_, 3), np.round(model.var_, 3)))

    # Question 2 - Empirically showing sample mean is consistent
    size = 100
    differences = np.empty(shape=size)
    x_axis = np.linspace(10, 1000, num=size, dtype=np.int64)
    for i, m in enumerate(x_axis):
        y = UnivariateGaussian().fit(samples[:m])
        differences[i] = np.abs(y.mu_ - mu)
    go.Figure(go.Scatter(x=x_axis, y=differences, mode='markers', name='Difference'),
              layout=dict(title="The Difference Between The Estimated And True Mean Value "
                                "As A Function Of The Number Of Samples",
                          xaxis_title="Number Of Samples",
                          yaxis_title="Difference",
                          height=600,
                          width=1000)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = model.pdf(samples)
    go.Figure(go.Scatter(x=samples, y=pdfs, mode='markers', name='Empirical PDF'),
              layout=dict(title="PDF Values Of A Fitted Model",
                          xaxis_title="Sample Value",
                          yaxis_title="PDF Value",
                          height=400,
                          width=900)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    cov = np.array([[1.0, 0.2, 0.0, 0.5],
                    [0.2, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.5, 0.0, 0.0, 1.0]])
    samples = np.random.multivariate_normal(mean=np.array([0.0, 0.0, 4.0, 0.0]),
                                            cov=cov,
                                            size=1000)
    model = MultivariateGaussian().fit(samples)
    print(np.round(model.mu_, 3))
    print(np.round(model.cov_, 3))

    # Question 5 - Likelihood evaluation
    size = 200
    axis = np.linspace(-10, 10, num=size, dtype=np.float16)
    values = np.empty(shape=(size, size))
    for i, a in enumerate(axis):
        for j, b in enumerate(axis):
            values[i, j] = MultivariateGaussian.log_likelihood(mu=np.array([a, 0, b, 0]),
                                                               cov=cov,
                                                               X=samples)
    go.Figure(go.Heatmap(x=axis, y=axis, z=values),
              layout=dict(title="The Log-Likelihood Of Multivariate Gaussian "
                                "As A Function Of The Mean Of The Features 1, 3",
                          xaxis_title="Sample Value",
                          yaxis_title="Sample Value",
                          height=900,
                          width=900)).show()

    # Question 6 - Maximum likelihood
    max_index = np.argmax(values)
    a, b = np.unravel_index(max_index, values.shape)
    x, y = np.round(axis[a], 3), np.round(axis[b], 3)
    print(x, y)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

from math import atan2, pi
from typing import Tuple

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.metrics import accuracy
from utils import *


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def log_loss(model: Perceptron, _: np.ndarray, __: int):
            losses.append(model.loss(X, y))

        Perceptron(callback=log_loss).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        training_iterations = list(range(len(losses)))
        title = f"Perceptron Training Error of {n} dataset"
        labels = {"x": "Iteration", "y": "Misclassification Error"}
        image_title = f"Q1_{n}.png"
        px.line(x=training_iterations, y=losses, title=title, labels=labels).write_image(image_title)


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        gnb = GaussianNaiveBayes().fit(X, y)
        lda_prediction = lda.predict(X)
        gnb_prediction = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        lda_accuracy = round(100 * accuracy(y, lda_prediction), )
        gnb_accuracy = round(100 * accuracy(y, gnb_prediction), )
        lda_title = f"LDA (accuracy = {lda_accuracy}%)"
        gnb_title = f"Gaussian Naive Bayes (accuracy = {gnb_accuracy}%)"
        fig = make_subplots(rows=1, cols=2, subplot_titles=(lda_title, gnb_title))

        # Add traces for data-points setting symbols and colors
        lda_marker = dict(color=gnb_prediction, symbol=class_symbols[y], colorscale=class_colors(3))
        gnb_marker = dict(color=lda_prediction, symbol=class_symbols[y], colorscale=class_colors(3))
        lda_scatter = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=lda_marker)
        gnb_scatter = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',marker=gnb_marker)
        fig.add_traces([lda_scatter, gnb_scatter], rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        marker = dict(symbol="x", color="black", size=15)
        lda_scatter_2 = go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers", marker=marker)
        gnb_scatter_2 = go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1], mode="markers", marker=marker)
        fig.add_traces([gnb_scatter_2, lda_scatter_2], rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(3):
            lda_ellipse = get_ellipse(lda.mu_[i], lda.cov_)
            gnb_ellipse = get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i]))
            fig.add_traces([lda_ellipse, gnb_ellipse], rows=[1, 1], cols=[1, 2])

        dataset = f[:-4]
        title = f"Comparing Gaussian Classifiers - {dataset} dataset"
        filename = f"LDA VS Naive Bayes {dataset}.png"
        fig.update_yaxes(scaleanchor="x", scaleratio=1) \
            .update_layout(title_text=title, showlegend=False) \
            .write_image(filename)


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

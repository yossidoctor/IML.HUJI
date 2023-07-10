import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    v_list = list()
    w_list = list()

    def callback(gradient_descent, weight, val, grad, t, eta, delta):
        w_list.append(weight)
        v_list.append(val)

    return callback, v_list, w_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    l1_vals = list()
    for eta in etas:
        callback, vals, weights = get_gd_state_recorder_callback()
        fixed_lr = FixedLR(eta)
        l1_mod = L1(init)
        l1_grad_descent = GradientDescent(fixed_lr, tol=1e-5, max_iter=1000, callback=callback)
        l1_grad_descent.fit(l1_mod, None, None)
        title = f"L1_Descent_Trajectory_with_eta_{eta}"
        plot_descent_path(L1, np.array(weights), title).write_image(title + ".png")
        l1_vals.append(vals)

    l2_values = list()
    for eta in etas:
        callback, vals, weights = get_gd_state_recorder_callback()
        fixed_lr = FixedLR(eta)
        l2_mod = L2(init)
        l2_grad_descent = GradientDescent(fixed_lr, tol=1e-5, max_iter=1000, callback=callback)
        l2_grad_descent.fit(l2_mod, None, None)
        title = f"L2_Descent_Trajectory_with_eta_{eta}"
        plot_descent_path(L2, np.array(weights), title).write_image(title + ".png")
        l2_values.append(vals)

    l1_fig = go.Figure().update_xaxes(title="iterations number"). \
        update_yaxes(title="norm values"). \
        update_layout(title="L1 Norms as a Function of Number of Iterations")
    l2_fig = go.Figure().update_xaxes(title="iterations number") \
        .update_yaxes(title="norm values"). \
        update_layout(title="L2 Norms as a Function of Number of Iterations")
    l1_iters = max([len(l1_val) for l1_val in l1_vals])
    l2_iters = max([len(l2_val) for l2_val in l2_values])
    for i in range(len(etas)):
        l1_fig.add_trace(go.Scatter(x=np.arange(l1_iters), y=np.array(l1_vals[i]),
                                    mode="markers+lines", showlegend=True,
                                    name=f"eta={etas[i]}"))
        l2_fig.add_trace(go.Scatter(x=np.arange(l2_iters), y=np.array(l2_values[i]),
                                    mode="markers+lines", showlegend=True,
                                    name=f"eta={etas[i]}"))
    l1_fig.write_image("l1_norms.png")
    l2_fig.write_image("l2_norms.png")

    # printing the lowest loss for each of the modules
    l1_best_losses = [min(eta_loss) for eta_loss in l1_vals]
    best_l1_loss = None
    best_l1_ind = None
    l2_best_losses = [min(eta_loss) for eta_loss in l2_values]
    best_l2_loss = None
    best_l2_ind = None
    for i, loss in enumerate(l1_best_losses):
        if best_l1_loss is None or loss < best_l1_loss:
            best_l1_loss = loss
            best_l1_ind = i
    for i, loss in enumerate(l2_best_losses):
        if best_l2_loss is None or loss < best_l2_loss:
            best_l2_loss = loss
            best_l2_ind = i
    print(f"L1 best loss {best_l1_loss} achieved with eta={etas[best_l1_ind]}")
    print(f"L2 best loss {best_l2_loss} achieved with eta={etas[best_l2_ind]}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    cb = get_gd_state_recorder_callback()[0]
    g_descent = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000, callback=cb)
    log_mod = LogisticRegression(solver=g_descent).fit(X_train.to_numpy(), y_train.to_numpy())
    fpr, tpr, thresholds = roc_curve(y_train, log_mod.predict_proba(X_train.to_numpy()))
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).write_image("FittedModelROC.png")
    best_a = round(thresholds[np.argmax(tpr - fpr)], 2)
    best_a_lr = LogisticRegression(solver=g_descent, alpha=best_a).fit(X_train.to_numpy(), y_train.to_numpy())
    print(f"Best alpha: {best_a}")
    print(f"Best alpha test error: {best_a_lr.loss(X_test.to_numpy(), y_test.to_numpy())}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for pen in ["l1", "l2"]:
        validation_errs = list()
        for lam in lambdas:
            curr_log_mod = LogisticRegression(solver=g_descent, penalty=pen, lam=lam, alpha=0.5)
            validation_errs.append(cross_validate(curr_log_mod, X_train.to_numpy(),
                                                  y_train.to_numpy(), misclassification_error)[1])
        best_lambda = lambdas[np.argmin(validation_errs)]
        loss = LogisticRegression(solver=g_descent, penalty=pen, lam=best_lambda, alpha=0.5) \
            .fit(X_train.to_numpy(), y_train.to_numpy()).loss(X_test.to_numpy(), y_test.to_numpy())
        print(f"{pen} best lambda is: {best_lambda} with test error of: {loss}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()

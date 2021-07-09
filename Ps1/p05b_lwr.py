import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval,y_eval=util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    # Fit a LWR model
    clf=LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)
    # Get MSE value on the validation set
    y_pred=clf.predict(x_eval)
    MSE=(y_eval-y_pred).dot(y_eval-y_pred)/(2*y_eval.shape[0])
    print ("MSE = ",MSE)
    # Plot validation predictions on top of training set
    plt.plot(x_train,y_train,'bx')
    plt.plot(x_eval,y_pred,'ro')
    # No need to save predictions
    # Plot data
    plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x=x
        self.y=y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        y=np.zeros(x.shape[0])

        for i in range(x.shape[0]):
            w=np.zeros(self.x.shape[0])
            for j in range(self.x.shape[0]):
                w[j]=np.exp(-((np.linalg.norm(self.x[j]-x[i]))**2)/(2*((self.tau)**2)))

            W=np.diag(w)

            theta=np.linalg.inv((self.x.transpose()).dot(W.dot(self.x))).dot((self.x.transpose()).dot(W.dot(self.y)))
            y[i]=x[i].dot(theta) 

        return y
        # *** END CODE HERE ***
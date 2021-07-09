import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval,y_eval=util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    clf = PoissonRegression(step_size=lr)
    # Fit a Poisson Regression model
    clf.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    np.savetxt(pred_path,clf.predict(x_eval))
    # *** END CODE HERE ***

class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def h(self,x):

        return np.exp(x.dot(self.theta))

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.theta_new=np.zeros(x.shape[1])
        self.theta=np.zeros(x.shape[1])
        while 1:

            self.theta_new=self.theta+(self.step_size/x.shape[0])*(y - self.h(x)).dot(x)
            
            if np.linalg.norm(self.theta_new -self.theta)<self.eps:
                self.theta=self.theta_new

                break
            else :
                self.theta=self.theta_new
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return self.h(x)
        # *** END CODE HERE ***



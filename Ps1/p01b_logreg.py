import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train,y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval,y_eval=util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    util.plot(x_train,y_train,clf.theta)
    util.plot(x_eval,y_eval,clf.theta)
    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path,clf.predict(x_eval))
    # *** END CODE HERE ***

def sig(x,theta):
    return 1/(1+np.exp(-np.dot(x,theta)))

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.theta=np.zeros((x.shape[1],1))

        while 1:
            h=sig(x,self.theta)
            y=np.reshape(y,(y.shape[0],1))
            gradient=np.dot(np.transpose(x),h-y)
            H=np.zeros((x.shape[1],x.shape[1]))

            for i in range(x.shape[0]):
                temp=x[i]
                temp=np.reshape(temp,(x.shape[1],1))
                H+=h[i][0]*(1-h[i][0])*(np.dot(temp,temp.transpose()))
            H=H*(1.0/x.shape[0])
            gradient=gradient*(1.0/x.shape[0])
            self.thetanew =self.theta -np.dot(np.linalg.inv(H),gradient)
            if np.linalg.norm(self.thetanew -self.theta)<self.eps:
                self.theta=self.thetanew
                break
            else :
                self.theta=self.thetanew
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return sig(x,self.theta)
        # *** END CODE HERE ***

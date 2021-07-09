import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval,y_eval=util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    # Train a GDA classifier
    clf=GDA()
    clf.fit(x_train, y_train)
    # Plot decision boundary on validation set
    x_train=util.add_intercept(x_train)
    util.plot(x_train,y_train,clf.theta)
    util.plot(x_eval,y_eval,clf.theta)
    # Use np.savetxt to save outputs from validation set to pred_path
    np.savetxt(pred_path,clf.predict(x_eval))
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***

        # Find phi, mu_0, mu_1, and sigma
        
        m=y.shape[0]
        phi=(float)(y[y==1].shape[0])/m
        mu_0=np.sum(x[y==0,:],axis=0)/y[y==0].shape[0]
        mu_1=np.sum(x[y==1,:],axis=0)/y[y==1].shape[0]      
        print (np.sum(x[y==0,:],axis=0))
        mu_0=np.reshape(mu_0,(mu_0.shape[0],1))
        mu_1=np.reshape(mu_1,(mu_1.shape[0],1))
        x_m=x
        x_m[y==0]=x_m[y==0]-mu_0.transpose()
        x_m[y==1]=x_m[y==1]-mu_1.transpose()
        sigma=np.dot(np.transpose(x_m),x_m)/m
        sigma_inv=np.linalg.inv(sigma)

        # Write theta in terms of the parameters
        self.theta_1=np.dot(sigma_inv,mu_1)-np.dot(sigma_inv,mu_0)      
        self.theta_0=0.5*(np.dot(np.transpose(mu_0),np.dot(sigma_inv,mu_0))-np.dot(np.transpose(mu_1),np.dot(sigma_inv,mu_1)))+np.log(phi/(1-phi))
        
        self.theta=np.zeros((self.theta_1.shape[0]+1,1))

        self.theta[0]=self.theta_0
        self.theta[1:]=self.theta_1

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        H= 1/(1+np.exp(-np.dot(x,self.theta)))
        H[H>=0.5]=1
        H[H<0.5]=0
        return H
        # *** END CODE HERE
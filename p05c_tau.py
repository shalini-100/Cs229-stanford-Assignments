import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid,y_valid=util.load_dataset(valid_path, add_intercept=True)
    x_test,y_test=util.load_dataset(test_path, add_intercept=True)
    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    MSE=[]
    for tau in tau_values:
        clf=LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)
        y_pred=clf.predict(x_valid)
        MSE.append((y_valid-y_pred).dot(y_valid-y_pred)/(2*y_valid.shape[0]))
        #print ("tau=",tau,"MSE=",MSE[-1])
        #plt.plot(x_train,y_train,'bx')
        #plt.plot(x_valid,y_pred,'ro')
        #plt.show()
    i=MSE.index(min(MSE))
    tau=tau_values[i]
    # Run on the test set to get the MSE value
    clf=LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_test)
    MSE_test=(y_test-y_pred).dot(y_test-y_pred)/(2*y_test.shape[0])
    print ("MSE = ",MSE_test)
    # Save predictions to pred_path
    np.savetxt(pred_path,y_pred)
    # Plot data
    plt.plot(x_train,y_train,'bx')
    plt.plot(x_test,y_pred,'ro')
    plt.show()
    # *** END CODE HERE ***

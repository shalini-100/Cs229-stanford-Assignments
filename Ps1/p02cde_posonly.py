import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels

    x_train,t_train = util.load_dataset(train_path,'t', add_intercept=True)
    x_test,t_test=util.load_dataset(test_path,'t',add_intercept=True)

    clf_t = LogisticRegression()
    clf_t.fit(x_train, t_train)
    predict_1=clf_t.predict(x_test)
    util.plot(x_test,t_test,clf_t.theta)
    np.savetxt(pred_path_c,predict_1)

    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels

    x_train,y_train = util.load_dataset(train_path,'y',add_intercept=True)
    x_test,t_test=util.load_dataset(test_path,'t', add_intercept=True)

    clf_y = LogisticRegression()
    clf_y.fit(x_train, y_train)
    predict_2=clf_y.predict(x_test)
    util.plot(x_test,t_test,clf_y.theta)
    np.savetxt(pred_path_d,predict_2)

    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels

    x_val,y_val=util.load_dataset(valid_path,'y', add_intercept=True)
    predict_3=clf_y.predict(x_val)
    alpha=np.sum(predict_3[y_val==1])/predict_3[y_val==1].shape[0]
    predict_2_new=predict_2/alpha
    util.plot(x_test,t_test,clf_y.theta,alpha)
    np.savetxt(pred_path_e,predict_2_new)

    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
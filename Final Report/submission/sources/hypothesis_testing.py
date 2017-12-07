from scipy.stats import t
import numpy as np


def hypothesis_testing(result_arr1, result_arr2):

    if result_arr1.size != result_arr2.size:
        print "Two arrays are of different sizes"
        return

    # Number of folds
    n = result_arr1.size

    # Mean of the two arrays
    mean1 = np.mean(result_arr1)
    mean2 = np.mean(result_arr2)

    # Variance of the two arrays
    var1 = np.var(result_arr1)
    var2 = np.var(result_arr2)

    # Calculating x
    x = ((mean1 - mean2) * np.sqrt(n)) / np.sqrt(var1 + var2)

    # Calculating degrees of freedom
    df = np.ceil((((var1 + var2) ** 2) * (n - 1)) / (var1 ** 2 + var2 ** 2))

    #print 'Degrees of freedom %d' %df

    alpha = 0.05 # Using 0.05 as significance level so confidence level is (1- alpha) which is 0.95

    # Calculating the Inverse CDF of x at (1 - alpha) and df for student's t-distribution
    x_alpha_df = t.ppf((1 - alpha), df);

    #print(x_alpha_df)
    #if x > x_alpha_df:
     #   print 'We reject the Null hypothesis'
    #else:
     #   print 'We can not reject the Null hypothesis'

    return x, x_alpha_df


def main():
    a = np.array([12, 10, 14, 13, 13, 10, 11, 10, 11])
    b = np.array([10, 8, 12, 10, 11, 9, 13, 7, 9])

    print 'Value of x and value of x at confidence level .95 and calculated degrees of freedom is (%f,%f)'\
          %hypothesis_testing(a,b)



if __name__ == '__main__':
    main()
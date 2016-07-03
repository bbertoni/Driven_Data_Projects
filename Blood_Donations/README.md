* **training_data.csv**: set of blood donation data used to train a machine learning algorithm, contains the features *patient number* (unlabeled), *Months since Last Donation*, *Number of Donations*, *Total Volume Donated* (c.c.), *Months since First Donation*, and *Made Donation in March 2007* (a binary variable: 0 if no, 1 if yes).

* **test_data.csv**: set of test blood donation data, contains the features *patient number* (unlabeled), *Months since Last Donation*, *Number of Donations*, *Total Volume Donated* (c.c.), and *Months since First Donation*.

* **Blood.py**: Python code that implements a basic logistic regression algorithm on **training_data.csv** to learn how to use the given features to classify whether or not a patient made a donation in March 2007.  This code then predicts the probability (between 0 and 1) that the patients in **test_data.csv** will make a donation in March 2007.

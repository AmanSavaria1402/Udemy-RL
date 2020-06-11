'''
    Plan of Action:
        1. Create Data that has a given relationship
        2. Make the SGD loop
                a. Get epoch count
                b. Get learning rate
                c. Get Batch Size
                d. Initialize the weight values
                e. Loop over datapoints 
                f. Update the grad value using the derivative for MSE
                g. Subtract the grad value from the old weight matrix
        3. Compare the error at the end of each epoch
        4. Plot the plot for the new and the old weights
'''


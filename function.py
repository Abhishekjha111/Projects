from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def metrices_calculation(ty,py):
    '''This function will calculate and display all the possible metrices.
    Inputs: true_y, pred_y
    Outputs: confusion metric, precision, recall, accuracy'''
    print('Accuracy:{}\n precision:{}\n recall:{}\n confusion matrix:{}'.format(accuracy_score(ty,py),
                                                                           precision_score(ty,py),
                                                                           recall_score(ty,py),
                                                                           confusion_matrix(ty,py)))
    
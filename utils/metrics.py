import numpy as np


class Evaluator(object):

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    def cal_f1_score(self):
        f1_scores = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            true_positive = self.confusion_matrix[i, i]
            false_positive = np.sum(self.confusion_matrix[:, i]) - true_positive
            false_negative = np.sum(self.confusion_matrix[i, :]) - true_positive

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

            f1_scores[i] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1_scores.mean()

    def cal_acc(self):
        true_positives = np.diag(self.confusion_matrix).sum()
        total_samples = self.confusion_matrix.sum()
        accuracy = true_positives / total_samples
        return accuracy
    

    def add_batch(self, pred, gt):
        self.confusion_matrix[gt, pred] += 1

    def reset(self):
            self.confusion_matrix = np.zeros((self.num_classes,) * 2)


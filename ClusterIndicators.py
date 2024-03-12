# -*- coding: utf-8 -*-

class ClusterIndicators:
  def __init__(self, pred, labels):
    self.pred = pred
    self.labels = labels
    
    true_positives, false_negatives, false_positives, true_negatives = confusion_matrix(pred, labels)
    self.true_positives = true_positives
    self.false_negatives = false_negatives
    self.false_positives = false_positives
    self.true_negatives = true_negatives
    
    self.precision = _calculate_precision(self)
    self.recall = _calculate_recall(self)
    self.f1 = _calculate_f1(self)

def confusion_matrix(pred , labels):
    true_positives, false_negatives, false_positives, true_negatives = 0, 0, 0, 0
    
    for i in range(0, len(labels), 1):
        for j in range(i, len(labels), 1):
        
            if i < j:
                if pred[i] == pred[j] and labels[i] == labels[j]:
                    true_positives += 1
            
                elif pred[i] != pred[j] and labels[i] != labels[j]:
                    false_negatives += 1
                
                elif pred[i] == pred[j] and labels[i] != labels[j]:
                    false_positives += 1
                
                elif pred[i] != pred[j] and labels[i] == labels[j]:
                    true_negatives += 1
    
    return true_positives, false_negatives , false_positives , true_negatives


def _calculate_precision(self):
    return self.true_positives / (self.true_positives + self.false_positives)

def _calculate_recall(self):
    return self.true_positives / (self.true_positives + self.false_negatives)    

def _calculate_f1(self):
    return 2 * self.precision * self.recall / (self.precision + self.recall)



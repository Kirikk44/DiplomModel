class TestMetrics:
    def __init__(self):
        pass
    @staticmethod
    def personalization(self, predicted_relevant, really_relevant):
        pass

    @staticmethod
    def recall(self, predicted_relevant, really_relevant):
        recall_sum = 0
        for i in range(len(predicted_relevant)):
            if predicted_relevant[i] in really_relevant:
                recall_sum += 1
        return recall_sum / len(really_relevant)

  # recall_sum (sum(recall_k))) / n
    @staticmethod
    def recall_average(self, predicted_relevant, really_relevant):
        total_len = len(predicted_relevant)
        recall_sum = 0
        for key, value in predicted_relevant.items():
        # для каждого требования
            if not value or key not in really_relevant or not really_relevant[key]:
                continue
            num_relevant = 0
            for doc in value:
                if doc in really_relevant[key]:
                    num_relevant += 1
            recall_sum += num_relevant / len(really_relevant[key])

        return recall_sum/total_len
    
    @staticmethod
    def precision_average(self, predicted_relevant, really_relevant):
        precision_sum = 0
        num_relevant = 0
        for key, value in predicted_relevant.items():
            if not value or key not in really_relevant or not really_relevant[key]:
                continue
            num_relevant = 0
            for doc in value:
                if doc in really_relevant[key]:
                    num_relevant += 1
            precision_sum += num_relevant / len(value)
        return precision_sum / len(predicted_relevant)

    @staticmethod
    def f1_average(self, predicted_relevant, really_relevant):
        pr = self.precision(predicted_relevant, really_relevant)
        rec = self.recall(predicted_relevant, really_relevant)

        return 2*pr*rec / (pr + rec)

    @staticmethod
    def mrr(self, predicted_relevant, really_relevant):
        pass

    @staticmethod    
    def auc(self, predicted_relevant, really_relevant):
        pass
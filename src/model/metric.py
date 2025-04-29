import math

class RecMetric:
    def __init__(self, k_list=(1, 5, 10)):
        self.k_list = k_list
        self.metric = {}
        self.reset_metric()

    def evaluate(self, preds, labels):
        for label in labels:
            pred_list = preds
            if label == -100:
                continue
            for k in self.k_list:
                self.metric[f'recall@{k}'] += float(self.compute_hit(pred_list, label, k)) # / len(labels)
                # self.metric[f'hit@{k}'] += self.compute_hit(pred_list, label, k)
                # self.metric[f'recall@{k}'] += self.compute_hit(pred_list, label, k) / len(labels)
                self.metric[f'ndcg@{k}'] += self.compute_ndcg(pred_list, label, k)
                self.metric[f'mrr@{k}'] += self.compute_mrr(pred_list, label, k)
            self.metric['count'] += 1
        # JP
        # self. metric['count'] += len(labels)
        # return self.compute_recall(pred_list, label, k)

    def compute_hit(self, pred_list, label, k):
        return int(label in pred_list[:k])
    
    def compute_mrr(self, pred_list, label, k):
        if label in pred_list[:k]:
            label_rank = pred_list.index(label)
            return 1 / (label_rank + 1)
        return 0

    def compute_ndcg(self, pred_list, label, k):
        if label in pred_list[:k]:
            label_rank = pred_list.index(label)
            return 1 / math.log2(label_rank + 2)
        return 0

    def reset_metric(self):
        for metric in ['recall', 'ndcg', 'mrr']:
            for k in self.k_list:
                self.metric[f'{metric}@{k}'] = 0
        self.metric['count'] = 0

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if k != 'count':
                report[k] = v / self.metric['count']
                print(k, " ", v, " ", self.metric['count'])
            else:
                report[k] = v
        return report

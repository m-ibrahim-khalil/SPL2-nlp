import numpy as np


class PostProcess:
    def __init__(self, context, p1, p2):
        self.context = context
        self.p1 = p1
        self.p2 = p2
        self.context_length = len(context)
        self.max_span_length = 10
        self.span_value_dict = {}
        self.spans = []
        self.exp = 0.3

    def getIndex(self):
        return np.argmax(self.p1), np.argmax(self.p2)

    def get_sorted_span(self):
        sorted_values = sorted(self.span_value_dict.values())  # Sort the values
        sorted_spans = []
        for i in sorted_values:
            for k in self.span_value_dict.keys():
                if self.span_value_dict[k] == i:
                    sorted_spans.append(k)
                    break
        return sorted_spans

    def get_matched(self, span1, span2):
        x1, y1 = span1
        x2, y2 = span2
        if (x1 <= x2 and y1 >= y2) or (x1 >= x2 and y1 <= y2):
            return 1.0
        if x1 <= x2 <= y1 <= y2:
            return (y1-x2+1)/(y1-x1+1)
        if x2 <= x1 <= y2 <= y1:
            return (y2-x1+1)/(y1-x1+1)
        else:
            totlen = max(y2, y1)-min(x2, x1)
            if totlen < 5:
                return 0.5
            else:
                return 0.1

    def get_best_five(self):
        sorted_spans = self.get_sorted_span()
        i = 0
        spans = []
        for span in reversed(sorted_spans):
            if i == 0:
                spans.append(span)
                i += 1
                continue
            flag = 1
            for key in spans:
                matched = self.get_matched(span, key)
                # print(matched, span, key)
                if matched > self.exp:
                    flag = 0
                    break
            if flag == 1:
                spans.append(span)
                # print(spans)
            i += 1
            if len(spans) == 5:
                break
        return spans

    def get_ans_span(self):
        for i, val1 in enumerate(self.p1[0]):
            for j, val2 in enumerate(self.p2[0]):
                if j > self.context_length - 1 or (j - i) >= self.max_span_length:
                    break
                if j < i:
                    continue
                self.span_value_dict[(i, j)] = val1*val2
        spans = self.get_best_five()
        return spans

    def postProcess(self):
        self.spans = self.get_ans_span()
        ans = []
        for span in self.spans:
            str = ''
            for i in range(span[0], span[1]+1):
                str += self.context[i] + ' '
            # print(str)
            ans.append(str)
        return ans


span_value_dict = {(1, 2): 5, (1, 3): 4, (1, 4): 7}

sorted_values = sorted(span_value_dict.values()) # Sort the values
sorted_spans = []

for i in sorted_values:
    for k in span_value_dict.keys():
        if span_value_dict[k] == i:
            sorted_spans.append(k)
            break

print(sorted_spans)


# first2pairs = {k: sorted_dict[k] for k in list(sorted_dict)[:2]}
#
# print(first2pairs)
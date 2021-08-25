import numpy as np

dir = "F://SPL2//preproccess_squad//"

context = np.load(dir+'numpy_data//context_data250_2.npy')
question = np.load(dir+'numpy_data//question_data250_2.npy')
output1 = np.load(dir+'numpy_data//output_data1_250_2.npy')
output2 = np.load(dir+'numpy_data//output_data2_250_2.npy')

# np.savetxt(dir+'csv_data//context250_2.csv', context, delimiter=',')
# np.savetxt(dir+'csv_data//question250_2.csv', question, delimiter=',')
# np.savetxt(dir+'csv_data//output1_250_2.csv', output1, delimiter=',')
# np.savetxt(dir+'csv_data//output2_250_2.csv', output2, delimiter=',')

with open(dir+'csv_data//context250_2.csv', 'wt', encoding='utf8') as context_file, \
     open(dir+'csv_data//question250_2.csv', 'wt', encoding='utf8') as question_file, \
     open(dir+'csv_data//output1_250_2.csv', 'wt', encoding='utf8') as out1_file,\
     open(dir+'csv_data//output2_250_2.csv', 'wt', encoding='utf8') as out2_file:

    np.savetxt(context_file, context, delimiter=',')
    np.savetxt(question_file, question, delimiter=',')
    np.savetxt(out1_file, output1, delimiter=',')
    np.savetxt(out2_file, output2, delimiter=',')


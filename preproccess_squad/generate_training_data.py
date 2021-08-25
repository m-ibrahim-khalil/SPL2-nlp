from tqdm import tqdm

directory = "F://SPL2//preproccess_squad//train_set//full_data//"

i = 0
with open(directory + 'train.context', 'r', encoding='utf8') as context_file, \
        open(directory + 'train.question', 'r', encoding='utf8') as question_file, \
        open(directory + 'train.span', 'r', encoding='utf8') as span_file,\
        open(directory + 'train.answer', 'r', encoding='utf8') as answer_file,\
        open(directory+'train250_20.context', 'w', encoding='utf8') as context_100,\
        open(directory+'train250_20.question', 'w', encoding='utf8') as question_100,\
        open(directory+'train250_20.span', 'w', encoding='utf8') as span_100,\
        open(directory+'train250_20.answer', 'w', encoding='utf8') as answer_100:

    for context, question, span, answer in zip(tqdm(context_file, total=86400), tqdm(question_file, total=86400),
                                       tqdm(span_file, total=86400), tqdm(answer_file, total=86400)):
        context1 = [context]
        question1 = [question]
        passage = [pas.strip() for pas in context1]
        cont = []
        for pas in passage:
            context_tokens = pas.split(" ")
            cont.append(context_tokens)
        original_passage = [pas.lower() for pas in passage]
        quest = []
        for ques in question1:
            question_tokens = ques.split(" ")
            quest.append(question_tokens)

        if len(quest[0]) > 20 or len(cont[0]) > 200:
            continue
        context_100.write(context)
        question_100.write(question)
        span_100.write(span)
        answer_100.write(answer)
        i = i+1
        print(i)

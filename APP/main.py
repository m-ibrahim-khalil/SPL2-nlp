import model.BiDAF as model
from layers.preprocess import Preprocess
from layers.postprocess import PostProcess
from tkinter import *
from tkinter import scrolledtext, messagebox


def get_level(root, txt, r, c, cspan=1, fontsize=12):
    label = Label(root,
                  text=txt,
                  font=("Times New Roman", fontsize)
                  )
    label.grid(column=c, row=r, columnspan=cspan, padx=5, pady=10)
    return label


def get_radio_btn(root, txt, var, val, func, r, c):
    radio = Radiobutton(root, text=txt, variable=var, value=val, command=func)
    radio.grid(row=r, column=c, padx=5, pady=10)
    return radio


def get_textfield(root, height, r):
    text_area = scrolledtext.ScrolledText(root,
                                          wrap=WORD,
                                          width=50,
                                          height=height,
                                          font=("Times New Roman", 15),
                                          bd=5
                                          )

    text_area.grid(column=0, row=r, pady=10, padx=20, columnspan=3)
    text_area.focus()
    return text_area


if __name__ == '__main__':
    bidef = model.BiDAF(400, 250, 20)
    root = Tk()
    root.title('QA App')
    root.iconbitmap('../images/2.ico')
    root.geometry('600x700+700+20')
    answer_list = ['abcdef']

    title = get_level(root, "Question Answer Model", 0, 0, 3, 16)
    context_title = get_level(root, 'Enter Context(within 250 words)', 2, 0, 3, 14)
    question_title = get_level(root, 'Enter Question(within 20 words)', 4, 0, 3, 14)
    answer_level = get_level(root, '', 6, 0, 3, 14)
    status = get_level(root, "Answer 0 of 0", 7, 0, 3, 14)  # bd=1, relief=SUNKEN, anchor=E

    var = IntVar()


    def func():
        # bidef = model.BiDAF(400, 250, 20)
        # file_name = f'../model/bidaf250_3{str(var.get())}.h5'
        # print(file_name)
        # bidef.load_bidaf(file_name)
        pass


    R1 = get_radio_btn(root, "Model-1", var, 0, func, 1, 0)
    R2 = get_radio_btn(root, "Model-2", var, 1, func, 1, 2)

    context_area = get_textfield(root, 10, 3)
    # context_area.insert(INSERT,
    #                     '''john had already begun to improve his channel forces before the loss of normandy and he rapidly built up further maritime capabilities after its collapse . most of these ships were placed along the cinque ports , but portsmouth was also enlarged . by the end of 1204 he had around 50 large galleys available ; another 54 vessels were built between 1209 and 1212 . william of wrotham was appointed " keeper of the galleys " , effectively john 's chief admiral . wrotham was responsible for fusing john 's galleys , the ships of the cinque ports and pressed merchant vessels into a single operational fleet . john adopted recent improvements in ship design , including new large transport ships called buisses and removable forecastles for use in combat .''')
    question_area = get_textfield(root, 2, 5)
    # question_area.insert(INSERT, '''who was appointed " keeper of the galleys ? " ''')
    submit_str = StringVar()


    def onSubmit():
        bidef = model.BiDAF(400, 250, 20)
        file_name = f'../model/bidaf250_3{str(var.get())}.h5'
        bidef.load_bidaf(file_name)
        global answer_level
        global button_forward
        global button_back
        answer_level.grid_forget()
        submit_str.set('Predicting....')
        context = context_area.get("1.0", END)
        question = question_area.get("1.0", END)
        if len(context) <= 1:
            messagebox.showwarning('Question Answering App', 'Context Field is Empty!')
            submit_str.set('Predict Answer')
            return
        if len(question) <= 1:
            messagebox.showwarning('Question Answering App', 'Question Field is Empty!')
            submit_str.set('Predict Answer')
            return

        process = Preprocess(context, question)
        c, q = process.processForModel()
        p1, p2 = bidef.predict(c, q)
        context = process.preprocess(context)
        answers = PostProcess(context, p1, p2).postProcess()
        answer_list[0] = answers
        answer_level = get_level(root, answers[0], 6, 0, 3, 14)
        print(answers)
        submit_str.set('Predict Answer')
        status = get_level(root, f'Answer 1 of {len(answers)}', 7, 0, 3, 14)
        button_forward = Button(root, text=">>", command=lambda: forward(2))
        button_back = Button(root, text="<<", command=back, state=DISABLED)
        button_back.grid(row=8, column=0, padx=5, pady=10)
        button_forward.grid(row=8, column=2, padx=5, pady=10)

    def forward(answer_number):
        global answer_level
        global button_forward
        global button_back
        answer_level.grid_forget()
        answer_level = get_level(root,
                                 answer_list[0][answer_number - 1],
                                 6, 0, 3, 14
                                 )
        button_forward = Button(root, text=">>", command=lambda: forward(answer_number + 1))
        button_back = Button(root, text="<<", command=lambda: back(answer_number - 1))

        if answer_number == 5:
            button_forward = Button(root, text=">>", state=DISABLED)

        button_back.grid(row=8, column=0, padx=5, pady=10)
        button_forward.grid(row=8, column=2, padx=5, pady=10)
        status = get_level(root, f"Answer {answer_number} of {len(answer_list[0])}", 7, 0, 3, 14)


    def back(answer_number):
        global answer_level
        global button_forward
        global button_back

        answer_level.grid_forget()
        answer_level = get_level(root,
                                 answer_list[0][answer_number - 1],
                                 6, 0, 3, 14
                                 )
        button_forward = Button(root, text=">>", command=lambda: forward(answer_number + 1))
        button_back = Button(root, text="<<", command=lambda: back(answer_number - 1))

        if answer_number == 1:
            button_back = Button(root, text="<<", state=DISABLED)

        button_back.grid(row=8, column=0, padx=5, pady=10)
        button_forward.grid(row=8, column=2, padx=5, pady=10)
        status = get_level(root, f"Answer {answer_number} of {len(answer_list[0])}", 7, 0, 3, 14)


    submit_btn = Button(root,
                        textvariable=submit_str,
                        command=lambda: onSubmit(),
                        font=("Bold", 14),
                        bg='#3D4C53',        #20bebe paste  #d64747 red
                        fg='white',
                        height=2,
                        width=15
                        )
    submit_str.set('Predict Answer')
    button_back = Button(root, text="<<", command=back, state=DISABLED, font=("Bold", 14),
                         bg='#3D4C53',
                         fg='white')
    button_forward = Button(root, text=">>", command=lambda: forward(2), font=("Bold", 14),
                            bg='#3D4C53',
                            fg='white')

    button_back.grid(row=8, column=0, padx=5, pady=10)
    button_forward.grid(row=8, column=2, padx=5, pady=10)
    submit_btn.grid(column=1, row=8, padx=5, pady=10)

    root.mainloop()

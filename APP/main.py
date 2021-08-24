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
    label.grid(column=c, row=r, columnspan=cspan)
    return label


def get_radio_btn(root, txt, var, val, func, r, c):
    radio = Radiobutton(root, text=txt, variable=var, value=val, command=func)
    radio.grid(row=r, column=c)
    return radio


def get_textfield(root, height, r):
    text_area = scrolledtext.ScrolledText(root,
                                          wrap=WORD,
                                          width=55,
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
    root.title('BiDAF Question Answering App')
    root.iconbitmap('../images/2.ico')
    root.geometry('600x1000+50+50')
    answer_list = ['answer']

    title = get_level(root, "Question Answer App", 0, 0, 3, 15)
    context_title = get_level(root, 'Enter Context(within 250 words)', 2, 0, 3, 12)
    question_title = get_level(root, 'Enter Question(within 20 words)', 4, 0, 3, 12)
    answer_level = get_level(root, '', 6, 0, 3)
    status = get_level(root, "Answer 0 of 0", 7, 0, 3, 12)  # bd=1, relief=SUNKEN, anchor=E

    var = IntVar()


    def func():
        file_name = f'../model/bidaf250_3{str(var.get())}.h5'
        print(file_name)
        bidef.load_bidaf(file_name)


    R1 = get_radio_btn(root, "Model-1", var, 0, func, 1, 0)
    R2 = get_radio_btn(root, "Model-2", var, 1, func, 1, 2)

    context_area = get_textfield(root, 10, 3)
    question_area = get_textfield(root, 2, 5)

    submit_str = StringVar()


    def onSubmit():
        global answer_level
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
        answer_level.grid_forget()
        answer_level = get_level(root, answers[0], 6, 0, 3)
        # print(answers)
        submit_str.set('Predict Answer')
        status = get_level(root, f'Answer 1 of {len(answers)}', 7, 0, 3, 12)


    def forward(answer_number):
        global answer_level
        global button_forward
        global button_back
        answer_level.grid_forget()
        answer_level = get_level(root,
                                 answer_list[0][answer_number - 1],
                                 6, 0, 3, 12
                                 )
        button_forward = Button(root, text=">>", command=lambda: forward(answer_number + 1))
        button_back = Button(root, text="<<", command=lambda: back(answer_number - 1))

        if answer_number == 5:
            button_forward = Button(root, text=">>", state=DISABLED)

        button_back.grid(row=8, column=0)
        button_forward.grid(row=8, column=2, pady=10)
        status = get_level(root, f"Answer {answer_number} of {len(answer_list[0])}", 7, 0, 3, 12)


    def back(answer_number):
        global answer_level
        global button_forward
        global button_back

        answer_level.grid_forget()
        answer_level = get_level(root,
                                 answer_list[0][answer_number - 1],
                                 6, 0, 3, 12
                                 )
        button_forward = Button(root, text=">>", command=lambda: forward(answer_number + 1))
        button_back = Button(root, text="<<", command=lambda: back(answer_number - 1))

        if answer_number == 1:
            button_back = Button(root, text="<<", state=DISABLED)

        button_back.grid(row=8, column=0)
        button_forward.grid(row=8, column=2, pady=10)
        status = get_level(root, f"Answer {answer_number} of {len(answer_list[0])}", 7, 0, 3, 12)


    submit_btn = Button(root,
                        textvariable=submit_str,
                        command=lambda: onSubmit(),
                        font=("Bold", 12),
                        bg='#20bebe',
                        fg='white',
                        height=2,
                        width=15
                        )
    submit_str.set('Predict Answer')
    button_back = Button(root, text="<<", command=back, state=DISABLED,
                         bg='#20bebe',
                         fg='white')
    button_forward = Button(root, text=">>", command=lambda: forward(2),
                            bg='#20bebe',
                            fg='white')

    button_back.grid(row=8, column=0)
    button_forward.grid(row=8, column=2, pady=10)
    submit_btn.grid(column=1, row=8)

    root.mainloop()

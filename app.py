from model import *
from summarize_from_model import *
# Core Packages
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import *
import tkinter.filedialog

# Structure and Layout
window = Tk()


window.title("Text Summarization Using Abstractive Method")
window.geometry("590x700")
window.config(background='yellow')

style = ttk.Style(window)
style.configure('lefttab.TNotebook', tabposition='wn')


# Functions


def get_summary():
    raw_text = str(entry.get('1.0', tk.END))
    final_text = summarize(raw_text)
    print(final_text)
    result = '\nSummary:{}'.format(final_text)
    window_display.insert(tk.END, result)


# Clear entry widget
def clear_text():
    entry.delete('1.0', END)


def clear_display_result():
    window_display.delete('1.0', END)


# MAIN TAB
l1 = Label(window, text="Enter Text To Summarize")
l1.grid(row=1, column=0)

entry = Text(window, height=15)
entry.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# BUTTONS

# BUTTONS
button1 = Button(window, text="Reset", command=clear_text,
                 width=12, bg='#e31717', fg='#fff')
button1.grid(row=4, column=0, padx=10, pady=10)

button2 = Button(window, text="Summarize", command=get_summary,
                 width=16, height=5, bg='#2de309', fg='#fff')
button2.grid(row=4, column=1, padx=10, pady=20)

button3 = Button(window, text="Clear Result",
                 command=clear_display_result, width=12, bg='#03A9F4', fg='#fff')
button3.grid(row=5, column=0, padx=10, pady=10)


# Display Screen For Result
window_display = Text(window, height=15)
window_display.grid(row=7, column=0, columnspan=3, padx=5, pady=5)


window.mainloop()

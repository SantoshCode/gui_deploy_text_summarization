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
window.title("Summaryzer GUI")
window.geometry("700x400")
window.config(background='black')

style = ttk.Style(window)
style.configure('lefttab.TNotebook', tabposition='wn',)


# TAB LAYOUT
tab_control = ttk.Notebook(window,style='lefttab.TNotebook')
 
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)
tab5 = ttk.Frame(tab_control)

# ADD TABS TO NOTEBOOK
tab_control.add(tab1, text=f'{"Home":^20s}')
tab_control.add(tab2, text=f'{"File":^20s}')

tab_control.add(tab5, text=f'{"About ":^20s}')


label1 = Label(tab1, text= 'Summaryzer',padx=5, pady=5)
label1.grid(column=0, row=0)
 
label2 = Label(tab2, text= 'File Processing',padx=5, pady=5)
label2.grid(column=0, row=0)


label3 = Label(tab4, text= 'Compare Summarizers',padx=5, pady=5)
label3.grid(column=0, row=0)

label4 = Label(tab5, text= 'About',padx=5, pady=5)
label4.grid(column=0, row=0)

tab_control.pack(expand=1, fill='both')

# Functions 
def get_summary():
	raw_text = str(entry.get('1.0',tk.END))
	final_text = summarize(raw_text)
	print(final_text)
	result = '\nSummary:{}'.format(final_text)
	tab1_display.insert(tk.END,result)


# Clear entry widget
def clear_text():
	entry.delete('1.0',END)

def clear_display_result():
	tab1_display.delete('1.0',END)


# Clear Text  with position 1.0
def clear_text_file():
	displayed_file.delete('1.0',END)

# Clear Result of Functions
def clear_text_result():
	tab2_display_text.delete('1.0',END)








# Functions for TAB 2 FILE PROCESSER
# Open File to Read and Process
def openfiles():
	file1 = tkinter.filedialog.askopenfilename(filetypes=(("Text Files",".txt"),("All files","*")))
	read_text = open(file1).read()
	displayed_file.insert(tk.END,read_text)


def get_file_summary():
	raw_text = displayed_file.get('1.0',tk.END)
	final_text = text_summarizer(raw_text)
	result = '\nSummary:{}'.format(final_text)
	tab2_display_text.insert(tk.END,result)




# MAIN NLP TAB
l1=Label(tab1,text="Enter Text To Summarize")
l1.grid(row=1,column=0)

entry=Text(tab1,height=10)
entry.grid(row=2,column=0,columnspan=2,padx=5,pady=5)

# BUTTONS
button1=Button(tab1,text="Reset",command=clear_text, width=12,bg='#03A9F4',fg='#fff')
button1.grid(row=4,column=0,padx=10,pady=10)

button2=Button(tab1,text="Summarize",command=get_summary, width=12,bg='#ced',fg='#fff')
button2.grid(row=4,column=1,padx=10,pady=10)

button3=Button(tab1,text="Clear Result", command=clear_display_result,width=12,bg='#03A9F4',fg='#fff')
button3.grid(row=5,column=0,padx=10,pady=10)

button4=Button(tab1,text="Main Points", width=12,bg='#03A9F4',fg='#fff')
button4.grid(row=5,column=1,padx=10,pady=10)

# Display Screen For Result
tab1_display = Text(tab1)
tab1_display.grid(row=7,column=0, columnspan=3,padx=5,pady=5)


#FILE PROCESSING TAB
l1=Label(tab2,text="Open File To Summarize")
l1.grid(row=1,column=1)

displayed_file = ScrolledText(tab2,height=7)# Initial was Text(tab2)
displayed_file.grid(row=2,column=0, columnspan=3,padx=5,pady=3)

# BUTTONS FOR SECOND TAB/FILE READING TAB
b0=Button(tab2,text="Open File", width=12,command=openfiles,bg='#c5cae9')
b0.grid(row=3,column=0,padx=10,pady=10)

b1=Button(tab2,text="Reset ", width=12,command=clear_text_file,bg="#b9f6ca")
b1.grid(row=3,column=1,padx=10,pady=10)

b2=Button(tab2,text="Summarize", width=12,command=get_file_summary,bg='blue',fg='#fff')
b2.grid(row=3,column=2,padx=10,pady=10)

b3=Button(tab2,text="Clear Result", width=12,command=clear_text_result)
b3.grid(row=5,column=1,padx=10,pady=10)

b4=Button(tab2,text="Close", width=12,command=window.destroy)
b4.grid(row=5,column=2,padx=10,pady=10)

# Display Screen
# tab2_display_text = Text(tab2)
tab2_display_text = ScrolledText(tab2,height=10)
tab2_display_text.grid(row=7,column=0, columnspan=3,padx=5,pady=5)

# Allows you to edit
tab2_display_text.config(state=NORMAL)



# BUTTONS
button1=Button(tab3,text="Reset", width=12,bg='#03A9F4',fg='#fff')
button1.grid(row=4,column=0,padx=10,pady=10)

button2=Button(tab3,text="Get Text", width=12,bg='#03A9F4',fg='#fff')
button2.grid(row=4,column=1,padx=10,pady=10)

button3=Button(tab3,text="Clear Result",width=12,bg='#03A9F4',fg='#fff')
button3.grid(row=5,column=0,padx=10,pady=10)

button4=Button(tab3,text="Summarize", width=12,bg='#03A9F4',fg='#fff')
button4.grid(row=5,column=1,padx=10,pady=10)


tab3_display_text = ScrolledText(tab3,height=10)
tab3_display_text.grid(row=10,column=0, columnspan=3,padx=5,pady=5)




variable = StringVar()
variable.set("SpaCy")
choice_button = OptionMenu(tab4,variable,"SpaCy","Gensim","Sumy","NLTK")
choice_button.grid(row=6,column=1)


# Display Screen For Result
tab4_display = ScrolledText(tab4,height=15)
tab4_display.grid(row=7,column=0, columnspan=3,padx=5,pady=5)


# About TAB
about_label = Label(tab5,text="Summaryzer GUI V.0.0.1 \n by Student of Kathmamdu University \n Department of Computer Engineering",pady=5,padx=5)
about_label.grid(column=0,row=1)

window.mainloop()


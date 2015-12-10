__author__ = 'EL13115'
import Tkinter
import tkFileDialog

import test

root = Tkinter.Tk()
filename = tkFileDialog.askopenfilename(parent=root, title='Choose a file')
help_class = test.main_args()
if filename != None:
    help_class.url = filename
    test.main(help_class)
else:
    test.main(help_class)

root.mainloop()

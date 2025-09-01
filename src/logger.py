"""
A simple logger that redirects stdout to a provided tkinter Text widget.
"""
import tkinter as tk
import sys

class RedirectedLogger:
    """A class to redirect stdout to a GUI text widget."""
    def __init__(self):
        self.text_widget = None
        self.old_stdout = sys.stdout

    def set_widget(self, text_widget):
        """Provide the tkinter widget to log to."""
        self.text_widget = text_widget

    def start(self):
        """Redirects stdout."""
        sys.stdout = self

    def stop(self):
        """Restores the original stdout."""
        sys.stdout = self.old_stdout

    def write(self, text):
        """This method is called by `print`."""
        self.old_stdout.write(text)
        if self.text_widget:
            try:
                self.text_widget.config(state='normal')
                self.text_widget.insert(tk.END, text)
                self.text_widget.see(tk.END) # Auto-scroll
                self.text_widget.config(state='disabled')
                self.text_widget.update_idletasks()
            except tk.TclError:
                self.text_widget = None

    def flush(self):
        """Flush method to satisfy the file-like object interface."""
        self.old_stdout.flush()

# Create a single, global instance
LOGGER = RedirectedLogger()
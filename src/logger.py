"""
A simple logger that redirects stdout to a tkinter Text widget.
Handles graceful application shutdown when the log window is closed.
"""
import tkinter as tk
from tkinter import scrolledtext
import sys

class RedirectedLogger:
    """A class to redirect stdout to a GUI window."""
    def __init__(self):
        self.log_window = None
        self.text_widget = None
        self.old_stdout = sys.stdout

    def start(self):
        """Creates the log window and redirects stdout."""
        if self.log_window is None:
            self.log_window = tk.Tk()
            self.log_window.title("Console Log")
            self.log_window.geometry("800x400")

            self.text_widget = scrolledtext.ScrolledText(self.log_window, wrap=tk.WORD, font=("Courier New", 9))
            self.text_widget.pack(expand=True, fill='both')
            self.text_widget.config(state='disabled')

            self.log_window.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.log_window.update()

        sys.stdout = self

    def _on_closing(self):
        """Called when the user clicks the 'X' button on the log window."""
        print("\nLog window closed. Terminating application...")
        self.stop()
        self.destroy()
        # A more forceful exit to ensure all matplotlib windows close too
        sys.exit(0)

    def stop(self):
        """Restores the original stdout."""
        sys.stdout = self.old_stdout

    def destroy(self):
        """Safely destroys the tkinter window if it exists."""
        if self.log_window:
            self.log_window.destroy()
            self.log_window = None

    def write(self, text):
        """This method is called by `print`. It appends text to the widget."""
        self.old_stdout.write(text)

        # Safely write to the widget only if the window still exists
        if self.log_window and self.text_widget:
            try:
                self.text_widget.config(state='normal')
                self.text_widget.insert(tk.END, text)
                self.text_widget.see(tk.END)
                self.text_widget.config(state='disabled')
                self.log_window.update_idletasks()
            except tk.TclError:
                # This can happen if the window is in the process of closing
                self.log_window = None # Mark window as gone

    def flush(self):
        """Flush method to satisfy the file-like object interface."""
        self.old_stdout.flush()

# Create a single, global instance of the logger
LOGGER = RedirectedLogger()

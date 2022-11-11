# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:30:14 2022

@author: PURUSHOT
"""

from tkinter import Button
from tkinter import Tk, Label
import json
import os, sys, subprocess

application_state = {"background": "white"}  
background_colors = ["white", "red", "blue", "yellow", "green"]


class main(Tk):
    def __init__(self):
       rootqwe = Tk()
       rootqwe.title("Test")
       rootqwe.geometry("400x400")
       reload_state(rootqwe)
       for color in background_colors:
          def change_background_wrapper(color=color, root=rootqwe):
             change_background(color, rootqwe)
          Button(rootqwe, text=color,command=change_background_wrapper).pack()
       rootqwe.mainloop()

def change_background(color, window):
   print("new color: " + color)
   window.configure(bg=color)
   application_state["background"] = color
   update_config()

def reload_state(window):
   config_file = open("config.json")
   conf = json.load(config_file)
   window.configure(bg=conf["background"])

def update_config():
   with open("config.json", "w") as conf:
      conf.write(json.dumps(application_state))

class Reloader(object):
    RELOADING_CODE = 3
    def start_process(self):
        """Spawn a new Python interpreter with the same arguments as this one,
        but running the reloader thread.
        """
        while True:
            print("starting Tkinter application...")
            args = [sys.executable] + sys.argv
            env = os.environ.copy()
            env['TKINTER_MAIN'] = 'true'
            exit_code = subprocess.call(args, env=env, close_fds=False)
            if exit_code != self.RELOADING_CODE:
                return exit_code

    def trigger_reload(self):
        self.log_reload()
        config_file = open("config.json")
        conf = json.load(config_file)
        object.configure(bg=conf["background"])
        sys.exit(self.RELOADING_CODE)

    def log_reload(self):
        print("reloading...")

def run_with_reloader(root, *hotkeys):
    """Run the given application in an independent python interpreter."""
    import signal
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
    reloader = Reloader()
    try:
        if os.environ.get('TKINTER_MAIN') == 'true':
            for hotkey in hotkeys:
                root.bind_all(hotkey, lambda event: reloader.trigger_reload())
            if os.name == 'nt':
                root.wm_state("iconic")
                root.wm_state("zoomed")
            root.mainloop()
        else:
            sys.exit(reloader.start_process())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
   run_with_reloader(main(), "<Control-R>", "<Control-r>")

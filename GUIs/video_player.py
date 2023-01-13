import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from pyglet.media import Source, Player


class VideoPlayerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Video Player")
        self.geometry("400x300")

        self.player = Player()
        self.source = Source('path/to/video.mp4')
        self.player.queue(self.source)

        self.time_label = ttk.Label(self, text="Time: 0:00")
        self.time_label.pack()

        self.play_button = ttk.Button(self, text="Play", command=self.play)
        self.play_button.pack()

        self.pause_button = ttk.Button(self, text="Pause", command=self.pause)
        self.pause_button.pack()

        self.stop_button = ttk.Button(self, text="Stop", command=self.stop)
        self.stop_button.pack()

        self.seek_button = ttk.Button(self, text="Seek", command=self.seek)
        self.seek_button.pack()

    def play(self):
        self.player.play()
        self.update_time()

    def pause(self):
        self.player.pause()

    def stop(self):
        self.player.stop()
        self.time_label.config(text="Time: 0:00")

    def seek(self):
        time = messagebox.askinteger("Seek", "Enter time in seconds:")
        self.player.seek(time)
        self.update_time()

    def update_time(self):
        time = self.player.time
        minutes = int(time / 60)
        seconds = int(time % 60)
        self.time_label.config(text=f"Time: {minutes}:{seconds}")
        self.after(1000, self.update_time)


if __name__ == '__main__':
    app = VideoPlayerApp()
    app.mainloop()

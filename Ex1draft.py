import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import Button



def flip(bin_num):
    return 1 - bin_num


class LifeGame:
    def __init__(self, size=8, live_prob=0.5, wraparound=False):
        self.change_history = []
        if size % 2 != 0:
            raise ValueError("Please even number N")
        self.size = size
        self.live_prob = live_prob
        self.wraparound = wraparound
        self.step_counter = 0
    
        #Buttons
        self.paused = True
        self.started = False

    def on_start(self, event):
        self.started = True
        self.paused = False
        if not self.started:
            self.state = np.random.random(size=(self.size, self.size))
            self.state[self.state < self.live_prob] = 0
            self.state[self.state >= self.live_prob] = 1

    def on_pause(self, event):
        self.paused = not self.paused

    def on_quit(self, event):
         plt.close()

    def on_toggle_wraparound(self, event):
        self.wraparound = not self.wraparound
        print(f"Wraparound is now {'ON' if self.wraparound else 'OFF'}")

    def step(self):
        prev_state = self.state.copy()
        i_start = 2 if not self.wraparound and self.step_counter % 2 == 0 else 0

        for i in range(i_start, self.size, 2):
            for j in range(i_start, self.size, 2):
                odd_mult = 1 if self.step_counter % 2 == 1 else -1
                count = self.state[i, j] + self.state[i + odd_mult, j] + self.state[i, j + odd_mult] + self.state[
                    i + odd_mult, j + odd_mult]
                self.rules(odd_mult, count, i, j)

        #  Measure change
        changed = np.sum(self.state != prev_state)
        percent_changed = (changed / self.state.size) * 100
        self.change_history.append(percent_changed)

    def rules(self, odd_mult, count, i, j):
        if count != 2:
            self.state[i, j] = flip(self.state[i, j])
            self.state[i + odd_mult, j] = flip(self.state[i + odd_mult, j])
            self.state[i, j + odd_mult] = flip(self.state[i, j + odd_mult])
            self.state[i + odd_mult, j + odd_mult] = flip(self.state[i + odd_mult, j + odd_mult])
            if count == 3:
                self.switch_places((i, j), (i + odd_mult, j + odd_mult))
                self.switch_places((i, j + odd_mult), (i + odd_mult, j))

    def switch_places(self, place1, place2):
        tmp = self.state[place1]
        self.state[place1] = self.state[place2]
        self.state[place2] = tmp

    def update_frame(self):
        if not self.started or self.paused or self.step_counter > self.max_steps:
            return

        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_aspect('equal')

        # Generate state if first run
        if self.step_counter == 0 and not hasattr(self, 'state'):
            self.state = np.random.random((self.size, self.size))
            self.state[self.state < self.live_prob] = 0
            self.state[self.state >= self.live_prob] = 1

        # Measure change (before stepping)
        prev = self.state.copy()
        self.step()
        changed = np.sum(self.state != prev)
        percent_change = (changed / self.state.size) * 100
        self.change_history.append(percent_change)

        # First draw grid
        self.ax.vlines(range(1, self.size, 2), 0, self.size, colors='r', linestyles='dashed', linewidth=0.5)
        self.ax.vlines(range(2, self.size, 2), 0, self.size, colors='blue', linestyles='solid', linewidth=0.5)
        self.ax.hlines(range(1, self.size, 2), 0, self.size, colors='r', linestyles='dashed', linewidth=0.5)
        self.ax.hlines(range(2, self.size, 2), 0, self.size, colors='blue', linestyles='solid', linewidth=0.5)

        # Then draw live cells (on top of the grid)
        if hasattr(self, 'state'):
            x, y = self.state.nonzero()
            self.ax.scatter(x=x + 0.5, y=y + 0.5, c='black', marker='s', s=5)

        # Title
        self.ax.set_title(f"Gen F{self.step_counter} | Δ {percent_change:.2f}%", fontsize=12)

        self.step_counter += 1
        self.fig.canvas.draw()

        # When done: show stability curve
        if self.step_counter > self.max_steps:
            self.show_stability_curve()

    def show_stability_curve(self):
        fig, ax = plt.subplots()
        ax.plot(self.change_history, label='Change % per step', color='blue')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Percent of Changed Cells')
        ax.set_title('Stability Curve')
        ax.grid(True)
        ax.legend()
        plt.show()

    def play(self, steps=250, pause_time=1):
        self.max_steps = steps
        self.pause_time = pause_time
        self.change_history = []

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        # Buttons
        ax_start = plt.axes([0.05, 0.05, 0.15, 0.075])
        btn_start = Button(ax_start, 'Start')
        btn_start.on_clicked(self.on_start)

        ax_pause = plt.axes([0.25, 0.05, 0.15, 0.075])
        btn_pause = Button(ax_pause, 'Pause')
        btn_pause.on_clicked(self.on_pause)

        ax_quit = plt.axes([0.45, 0.05, 0.15, 0.075])
        btn_quit = Button(ax_quit, 'Quit')
        btn_quit.on_clicked(self.on_quit)

        ax_wrap = plt.axes([0.65, 0.05, 0.25, 0.075])
        btn_wrap = Button(ax_wrap, 'Wraparound OFF')
        btn_wrap.on_clicked(lambda event: [
            self.on_toggle_wraparound(event),
            btn_wrap.label.set_text(f"Wraparound {'ON' if self.wraparound else 'OFF'}")
        ])

        # Timer-based animation
        self.timer = self.fig.canvas.new_timer(interval=int(pause_time * 1000))
        self.timer.add_callback(self.update_frame)
        self.timer.start()

        plt.show()

        # exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Ex1Bio',
        description='Executes the Ex1 program')
    parser.add_argument('-n', '--size', default=100, type=int)
    parser.add_argument('-s', '--steps', default=250, type=int)
    parser.add_argument('-p', '--proba', default=0.5, type=float)
    parser.add_argument('-w', '--wraparound', action='store_true', default=False)
    parser.add_argument('-t', '--pausetime', default=1, type=int)
    args = parser.parse_args()
    print(args)
    lg = LifeGame(size=args.size, live_prob=args.proba, wraparound=args.wraparound)
    lg.play(args.steps,args.pausetime)

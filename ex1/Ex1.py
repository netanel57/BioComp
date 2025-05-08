import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox


def flip(bin_num):
    return 1 - bin_num


class LifeGame:
    def __init__(self, size=8, live_prob=0.5, wraparound=False):
        self.change_history = []
        self.live_ratio_history = []
        self.dead_zone_history = []
        self.lifespan = None
        self.avg_lifespan_history = []
        self.max_lifespan_history = []
        self.first_stable_gen = None
        self.STABILITY_THRESHOLD = 0.5  # %change to consider stable

        if size % 2 != 0:
            raise ValueError("Please even number N")
        self.size = size
        self.live_prob = live_prob
        self.wraparound = wraparound
        self.step_counter = 0

        # Buttons
        self.paused = True
        self.started = False

        self.speed_mode = "medium"
        self.speed_map = {
            "slow": 1000,
            "medium": 300,
            "fast": 50
        }

    def on_speed_toggle(self, event):
        modes = list(self.speed_map.keys())
        current_idx = modes.index(self.speed_mode)
        next_idx = (current_idx + 1) % len(modes)
        self.speed_mode = modes[next_idx]
        new_interval = self.speed_map[self.speed_mode]
        self.timer.stop()
        self.timer = self.fig.canvas.new_timer(interval=new_interval)
        self.timer.add_callback(self.update_frame)
        self.timer.start()
        self.btn_speed.label.set_text(f"Speed: {self.speed_mode.capitalize()}")
        print(f"Speed set to {self.speed_mode} ({new_interval} ms/frame)")

    def on_start(self, event):
        if not self.started:
            # Read user input from TextBox
            try:
                val = int(self.gen_input.text)
                if val > 0:
                    self.max_steps = val
                else:
                    print("Please enter a positive number of generations.")
            except ValueError:
                print("Invalid number of generations. Using default:", self.max_steps)

            self.state = np.random.random((self.size, self.size))
            self.state[self.state < self.live_prob] = 0
            self.state[self.state >= self.live_prob] = 1

        self.lifespan = np.zeros((self.size, self.size))
        self.started = True
        self.paused = False
        self.timer.start()

    def on_pause(self, event):
        self.paused = not self.paused

    def on_quit(self, event):
        plt.close()

    def on_reset(self, event):
        self.paused = True
        self.started = False
        self.step_counter = 0
        self.change_history.clear()
        self.live_ratio_history.clear()
        self.dead_zone_history.clear()
        self.avg_lifespan_history.clear()
        self.max_lifespan_history.clear()
        self.first_stable_gen = None

        #
        if hasattr(self, 'state'):
            del self.state

        # Stop the timer
        if hasattr(self, 'timer'):
            self.timer.stop()

        # Reset the generation input
        if hasattr(self, 'gen_input'):
            self.gen_input.set_val(str(self.max_steps))

        # Reset the prob input
        if hasattr(self, 'prob_input'):
            self.prob_input.set_val(str(self.live_prob))

        # Clear the plot
        self.ax.clear()
        self.ax.set_title("Click 'Start' to begin a new simulation")
        self.fig.canvas.draw()

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

        # Measure change
        changed = np.sum(self.state != prev_state)
        percent_changed = (changed / self.state.size) * 100
        self.change_history.append(percent_changed)

        unchanged_cell = np.sum(self.state == prev_state)
        deadzone_cell = np.sum(unchanged_cell / self.state.size) * 100
        self.dead_zone_history.append(deadzone_cell)

    def show_dead_zone(self):
        fig, ax = plt.subplots()
        ax.plot(range(len(self.dead_zone_history)), self.dead_zone_history, label="Dead Zone", color="red")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Dead Zone ratio")
        ax.set_title("Dead Zone Ratio")
        ax.grid(True)
        ax.legend()
        plt.show()

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

    def set_prob(self, prob):
        self.live_prob = prob
        print(f"Live probability set to {prob}")

        if hasattr(self, 'prob_label'):
            self.prob_label.set_text(f"Live Probability: {prob}")
            self.fig.canvas.draw_idle()

    def update_frame(self):
        if not self.started or self.paused or self.step_counter >= self.max_steps:
            return

        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_aspect('equal')

        if self.step_counter == 0 and not hasattr(self, 'state'):
            self.state = np.random.random((self.size, self.size))
            self.state[self.state < self.live_prob] = 0
            self.state[self.state >= self.live_prob] = 1

        prev = self.state.copy()
        self.step()  # update state

        self.lifespan[self.state == 1] += 1  # Increase lifespan where cells are alive
        self.lifespan[self.state == 0] = 0  # Reset lifespan where cells died
        avg_life = np.mean(self.lifespan[self.state == 1]) if np.any(self.state == 1) else 0
        max_life = np.max(self.lifespan)

        self.avg_lifespan_history.append(avg_life)
        self.max_lifespan_history.append(max_life)

        # Change and live stats
        changed = np.sum(self.state != prev)
        percent_change = (changed / self.state.size) * 100
        if self.first_stable_gen is None and percent_change < self.STABILITY_THRESHOLD:
            self.first_stable_gen = self.step_counter

        live_ratio = np.sum(self.state) / self.state.size
        self.live_ratio_history.append(live_ratio * 100)
        self.change_history.append(percent_change)

        # Draw the grid
        self.ax.vlines(range(1, self.size, 2), 0, self.size, colors='r', linestyles='dashed', linewidth=0.5)
        self.ax.vlines(range(2, self.size, 2), 0, self.size, colors='blue', linestyles='solid', linewidth=0.5)
        self.ax.hlines(range(1, self.size, 2), 0, self.size, colors='r', linestyles='dashed', linewidth=0.5)
        self.ax.hlines(range(2, self.size, 2), 0, self.size, colors='blue', linestyles='solid', linewidth=0.5)

        x, y = self.state.nonzero()
        self.ax.scatter(x=x + 0.5, y=y + 0.5, c='black', marker='s', s=5)
        self.ax.set_title(f"Gen F{self.step_counter} | Δ {percent_change:.2f}%", fontsize=12)

        self.step_counter += 1
        self.fig.canvas.draw()

        if self.step_counter >= self.max_steps:
            self.show_stability_curve()
            self.show_dead_zone()
            self.show_lifespan_graph()

    def show_stability_curve(self):
        fig, ax = plt.subplots()

        generations = range(self.step_counter)  # Only up to current generation
        ax.plot(generations, self.change_history[:self.step_counter], label='Δ % Cells', color='blue')
        ax.plot(generations, self.live_ratio_history[:self.step_counter], label='Live Cell %', color='green')
        if self.first_stable_gen is not None:
            ax.axvline(self.first_stable_gen, color='gray', linestyle='--', label='Stabilization Point')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Percent')
        ax.set_title(f'Stability Curve ({self.step_counter} generations)')
        ax.grid(True)
        print("\n--- Simulation Summary ---")
        print(f"Stabilized at Generation: {self.first_stable_gen if self.first_stable_gen is not None else 'Never'}")
        print(f"Stability Threshold: {self.STABILITY_THRESHOLD}%")
        ax.legend()
        plt.show()

    def show_lifespan_graph(self):
        fig, ax = plt.subplots()
        ax.plot(range(len(self.avg_lifespan_history)), self.avg_lifespan_history, label='Avg Live Cell Lifespan',
                color='purple')
        ax.plot(range(len(self.max_lifespan_history)), self.max_lifespan_history, label='Max Lifespan', color='orange')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Lifespan (steps)')
        ax.set_title('Cell Lifespan Tracking')
        ax.grid(True)
        ax.legend()
        plt.show()

    def play(self, steps=250, pause_time=1):
        self.max_steps = steps
        self.pause_time = pause_time
        self.change_history = []

        self.fig, self.ax = plt.subplots()
        self.prob_label = self.ax.text(
            1, self.size - 3, "",
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
        )
        plt.subplots_adjust(bottom=0.2)

        # Buttons    #  reminder -[left/right, bottom/up, width, height]
        ax_start = plt.axes([0.05, 0.05, 0.15, 0.075])
        btn_start = Button(ax_start, 'Start', color='orange')
        btn_start.on_clicked(self.on_start)

        ax_pause = plt.axes([0.25, 0.05, 0.15, 0.075])
        btn_pause = Button(ax_pause, 'Pause', color='orange')
        btn_pause.on_clicked(self.on_pause)

        ax_quit = plt.axes([0.45, 0.05, 0.15, 0.075])
        btn_quit = Button(ax_quit, 'Quit', color='orange')
        btn_quit.on_clicked(self.on_quit)

        ax_wrap = plt.axes([0.65, 0.05, 0.25, 0.075])
        btn_wrap = Button(ax_wrap, 'Wraparound OFF', color='blue')
        btn_wrap.on_clicked(lambda event: [
            self.on_toggle_wraparound(event),
            btn_wrap.label.set_text(f"Wraparound {'ON' if self.wraparound else 'OFF'}")
        ])

        ax_reset = plt.axes([0.90, 0.90, 0.1, 0.05])
        btn_reset = Button(ax_reset, 'Reset', color='red')
        btn_reset.on_clicked(self.on_reset)

        # Text field for live probability
        ax_prob = plt.axes([0.3, 0.90, 0.1, 0.05])
        self.prob_input = TextBox(ax_prob, 'Live Prob (0-1):', initial=str(self.live_prob))

        def submit_prob(text):
            try:
                prob = float(text)
                if 0 <= prob <= 1:
                    self.set_prob(prob)
                else:
                    print("Please enter a probability between 0 and 1.")
            except ValueError:
                print("Invalid input. Please enter a numeric value between 0 and 1.")

        self.prob_input.on_submit(submit_prob)

        ax_speed = plt.axes([0.73, 0.90, 0.17, 0.05])
        self.btn_speed = Button(ax_speed, 'Speed: Medium')
        self.btn_speed.on_clicked(self.on_speed_toggle)
        self.timer = self.fig.canvas.new_timer(interval=self.speed_map[self.speed_mode])

        # Text field for number of generations
        ax_gen = plt.axes([0.3, 0.95, 0.1, 0.05])
        self.gen_input = TextBox(ax_gen, 'Number of generations:', initial=str(self.max_steps))

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
    lg.play(args.steps, args.pausetime)

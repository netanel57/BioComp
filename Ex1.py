import numpy as np
import matplotlib.pyplot as plt
import argparse


def flip(bin_num):
    return 1 - bin_num


class LifeGame:
    def __init__(self, size=8, live_prob=0.5, wraparound=False):
        if size % 2 != 0:
            raise ValueError("Please even number N")
        self.size = size
        self.live_prob = live_prob
        self.wraparound = wraparound
        self.step_counter = 0
        # generate initial state
        self.state = np.random.random(size=(size, size))
        self.state[self.state < live_prob] = 0
        self.state[self.state >= live_prob] = 1

    def step(self):
        i_start = 2 if not self.wraparound and self.step_counter % 2 == 0 else 0
        for i in range(i_start, self.size, 2):
            for j in range(i_start, self.size, 2):
                # window
                odd_mult = 1 if self.step_counter % 2 == 1 else -1

                count = self.state[i, j] + self.state[i + odd_mult, j] + self.state[i, j + odd_mult] + self.state[
                    i + odd_mult, j + odd_mult]
                self.rules(odd_mult, count, i, j)

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

    def play(self, steps=250, pause_time=1):
        for i in range(steps+1):
            plt.clf()
            plt.xlim(0, self.size)
            plt.ylim(0, self.size)
            plt.gca().set_aspect('equal')

            # Get transformation from data to pixels
            transform = plt.gca().transData
            p1 = transform.transform((0, 0))
            p2 = transform.transform((1, 1))
            cell_width = abs(p2[0] - p1[0])
            cell_height = abs(p2[1] - p1[1])

            # Convert to points
            dpi = plt.gcf().dpi
            cell_width_pt = cell_width * 72 / dpi
            cell_height_pt = cell_height * 72 / dpi
            line_width_x = 0.1 * cell_width_pt
            line_width_y = 0.1 * cell_height_pt
            cell_size_pts = min(cell_width_pt, cell_height_pt)
            marker_size = cell_size_pts ** 2

            plt.vlines(range(1, self.size, 2), 0, self.size, colors='r', linestyles='dashed', linewidth=line_width_x)
            plt.vlines(range(2, self.size, 2), 0, self.size, colors='blue', linestyles='solid', linewidth=line_width_x)
            plt.hlines(range(1, self.size, 2), 0, self.size, colors='r', linestyles='dashed', linewidth=line_width_y)
            plt.hlines(range(2, self.size, 2), 0, self.size, colors='blue', linestyles='solid', linewidth=line_width_y)
            x, y = self.state.nonzero()
            plt.scatter(x=x + 0.5, y=y + 0.5, c='black', marker='s', s=marker_size)
            plt.title(f'Step: {self.step_counter}')
            plt.pause(pause_time)
            self.step_counter += 1
            self.step()
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

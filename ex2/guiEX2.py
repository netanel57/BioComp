import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from Ex2 import GeneticAlgorithm, MagicSquareProblem

def run_algorithm():
    try:
        n = int(entry_n.get())
        generations = int(entry_gen.get())
        mutation_rate = float(entry_mut.get())
        variant = variant_var.get()

        if variant == "standard":
            learning_type = None
        elif variant == "darwinian":
            learning_type = "darwinian"
        elif variant == "lamarckian":
            learning_type = "lamarkian"
        else:
            learning_type = None

        ga = GeneticAlgorithm(
            MagicSquareProblem,
            problem_args={'size': n},
            elitism=2,
            crossover_points=4,
            mutation_rate=mutation_rate,
            learning_type=learning_type,
            learning_cap=n,
            population_seeds=np.arange(42, 42+100),
            pop_size=100,
            seed=32
        )

        best, best_fitness = ga.play(max_steps=generations)

        print("Best Magic Square:")
        print(best.square)

        fig, ax = plt.subplots()
        ax.set_title(f"Best Fitness: {best_fitness}")
        ax.axis('off')
        table = ax.table(cellText=best.square, loc='center', cellLoc='center')
        table.scale(1, 2)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

root = tk.Tk()
root.title("Magic Square Genetic Algorithm")

frame = tk.Frame(root, padx=10, pady=10)
frame.grid(row=0, column=0)

tk.Label(frame, text="Square size (N):").grid(row=0, column=0, sticky="w")
entry_n = tk.Entry(frame)
entry_n.insert(0, "5")
entry_n.grid(row=0, column=1)

tk.Label(frame, text="Generations:").grid(row=1, column=0, sticky="w")
entry_gen = tk.Entry(frame)
entry_gen.insert(0, "500")
entry_gen.grid(row=1, column=1)

tk.Label(frame, text="Mutation rate:").grid(row=2, column=0, sticky="w")
entry_mut = tk.Entry(frame)
entry_mut.insert(0, "0.05")
entry_mut.grid(row=2, column=1)

tk.Label(frame, text="Algorithm variant:").grid(row=3, column=0, sticky="w")
variant_var = tk.StringVar()
variant_combo = ttk.Combobox(frame, textvariable=variant_var)
variant_combo["values"] = ["standard", "darwinian", "lamarckian"]
variant_combo.current(0)
variant_combo.grid(row=3, column=1)

run_button = tk.Button(frame, text="Run", command=run_algorithm)
run_button.grid(row=4, columnspan=2, pady=10)

root.mainloop()

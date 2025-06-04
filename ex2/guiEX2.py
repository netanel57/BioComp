
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from Ex2 import GeneticAlgorithm, MagicSquareProblem
import tkinter.messagebox as messagebox



class MagicSquareApp:
    def __init__(self, master):
        self.master = master
        master.title("Magic Square Genetic Algorithm")  # woho name

        self.running = False

        # Main
        self.main_frame = tk.Frame(master, padx=10, pady=10)
        self.main_frame.pack()

        self.square_frame = tk.Frame(self.main_frame)
        self.square_frame.grid(row=0, column=0, padx=10)

        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.grid(row=0, column=1, padx=10, sticky="n")

        # --- Inputs ---
        row_counter = 0

        tk.Label(self.control_frame, text="Square size (N):").grid(row=row_counter, column=0, sticky="w")
        self.entry_n = tk.Entry(self.control_frame)
        self.entry_n.insert(0, "5")
        self.entry_n.grid(row=row_counter, column=1)
        row_counter += 1

        tk.Label(self.control_frame, text="Generations:").grid(row=row_counter, column=0, sticky="w")
        self.entry_gen = tk.Entry(self.control_frame)
        self.entry_gen.insert(0, "500")
        self.entry_gen.grid(row=row_counter, column=1)
        row_counter += 1

        tk.Label(self.control_frame, text="Mutation rate:").grid(row=row_counter, column=0, sticky="w")
        self.entry_mut = tk.Entry(self.control_frame)
        self.entry_mut.insert(0, "0.05")
        self.entry_mut.grid(row=row_counter, column=1)
        row_counter += 1

        #Square Type Selectio
        tk.Label(self.control_frame, text="Square type:").grid(row=row_counter, column=0, sticky="w")
        self.square_type_var = tk.StringVar(value="standard")
        radio_standard = tk.Radiobutton(self.control_frame, text="Standard", variable=self.square_type_var, value="standard")
        radio_perfect = tk.Radiobutton(self.control_frame, text="Most Perfect", variable=self.square_type_var, value="most_perfect")
        radio_standard.grid(row=row_counter, column=1, sticky="w")
        row_counter += 1
        radio_perfect.grid(row=row_counter, column=1, sticky="w")
        row_counter += 1

        tk.Label(self.control_frame, text="Evolution type:").grid(row=row_counter, column=0, sticky="w")
        self.variant_var = tk.StringVar()
        self.variant_combo = ttk.Combobox(self.control_frame, textvariable=self.variant_var)
        self.variant_combo["values"] = ["No additional constraints", "darwinian", "lamarckian"]
        self.variant_combo.current(0)
        self.variant_combo.grid(row=row_counter, column=1)
        row_counter += 1

        button_frame = tk.Frame(self.control_frame)
        button_frame.grid(row=row_counter, column=0, columnspan=2, pady=(10, 10))

        self.run_button = tk.Button(button_frame, text="Run", command=self.run_algorithm)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_algorithm)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset_ui)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.labels = []

    # it might look stuck, but it is what it is;
    def update_square_display(self, square,solved=False):
        self.clear_display()
        n = square.shape[0]
        for i in range(n):
            row_labels = []
            for j in range(n):
                value = square[i][j]
                color = "green" if solved else "black"
                label = tk.Label(
                    self.square_frame,
                    text=str(value),
                    width=4, height=2,
                    borderwidth=1, relief="solid",
                    font=("Courier", 12),
                    fg=color
                )
                label.grid(row=i, column=j, padx=1, pady=1)
                row_labels.append(label)
            self.labels.append(row_labels)
  #if reset
    def clear_display(self):
        for row in self.labels:
            for label in row:
                label.destroy()
        self.labels = []

    def run_algorithm(self):
        try:
            self.running = True
            self.clear_display()

            n = int(self.entry_n.get())
            generations = int(self.entry_gen.get())
            mutation_rate = float(self.entry_mut.get())
            square_mode = self.square_type_var.get()
            variant = self.variant_var.get()

            if variant == "darwinian":
                learning_type = "darwinian"
            elif variant == "lamarckian":
                learning_type = "lamarkian"
            else:
                learning_type = None

            # Determine square type
            if variant == "most_perfect":
                square_mode = "most_perfect"
            else:
                square_mode = "standard"

            ga = GeneticAlgorithm(
                MagicSquareProblem,
                problem_args={'size': n, 'mode': square_mode},
                elitism=2,
                crossover_points=4,
                mutation_rate=mutation_rate,
                learning_type=learning_type,
                learning_cap=n,
                population_seeds=np.arange(42, 42+100),
                pop_size=100,
                seed=32
            )

            best_fitness = float('inf')
            best_individual = None

            for gen in range(generations):
                if not self.running:
                    return

                ga.population = ga.learning_step(ga.population)
                ga.population = ga.generation_step(ga.population)
                curr = min(ga.population)
                if curr.fitness() < best_fitness:
                    best_fitness = curr.fitness()
                    best_individual = curr
                    self.update_square_display(best_individual.square)
                self.master.update()

            if best_individual:
                fig, ax = plt.subplots()
                ax.set_title(f"Best Fitness: {best_fitness}")
                ax.axis('off')
                table = ax.table(cellText=best_individual.square, loc='center', cellLoc='center')
                table.scale(1, 2)
                plt.show()
                messagebox.showinfo("Done", "Algorithm finished running.")


        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")


    def stop_algorithm(self):
        self.running = False

    def reset_ui(self):
        self.running = False
        self.clear_display()
        self.entry_n.delete(0, tk.END)
        self.entry_n.insert(0, "5")
        self.entry_gen.delete(0, tk.END)
        self.entry_gen.insert(0, "500")
        self.entry_mut.delete(0, tk.END)
        self.entry_mut.insert(0, "0.05")
        self.variant_combo.current(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = MagicSquareApp(root)
    root.mainloop()

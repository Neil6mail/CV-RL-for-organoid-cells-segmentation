import os
import csv
import tempfile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button,TextBox, RadioButtons
from computer_vision_algo import *
from image_treatment import *
from parameters import ALGO_REGISTRY
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox,filedialog,simpledialog
from PIL import Image, ImageTk
import platform
import uuid
import subprocess
from io import BytesIO



# Temp folder
TEMP_DIR = tempfile.mkdtemp(prefix="vision_tmp_")

# Évaluation humaines stockée ici
human_scores = {}

# Interface interactive
class AlgoViewer_human:
    def __init__(self, image_list, name_list):
        for i in range(len(name_list)):
            name_list[i] = f"{i} : " + name_list[i]

        self.images = image_list
        self.names = name_list
        self.index = 0

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.3)

        
        self.image_disp = self.ax.imshow(self.images[self.index], cmap='gray')
        self.ax.set_title("Chargement...")
        self.ax.axis('off')

        # Zones d’étoiles
        self.star_axes = []
        for i in range(2):
            ax_star = self.fig.add_axes([0.1, 0.18 - 0.07 * i, 0.8, 0.05])
            self.star_axes.append(ax_star)
            ax_star.axis('off')

        # Boutons
        self.axprev = self.fig.add_axes([0.15, 0.02, 0.15, 0.06])
        self.axnext = self.fig.add_axes([0.7, 0.02, 0.15, 0.06])
        self.bnext = Button(self.axnext, 'Suivant')
        self.bprev = Button(self.axprev, 'Précédent')
        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.update()

    def update(self):
        name = self.names[self.index]
        img = self.images[self.index]
        self.image_disp.set_data(img)
        self.ax.set_title(name)
        self.update_scores_display()
        self.fig.canvas.draw_idle()

    def update_scores_display(self):
        for i, ax in enumerate(self.star_axes):
            ax.clear()
            ax.set_xlim(0, 4.5)
            ax.set_ylim(0, 1)
            ax.axis("off")

            key = self.names[self.index]
            score = human_scores.get(key, {"clarte": 0, "contraste": 0})
            current = score["clarte"] if i == 0 else score["contraste"]

            positions = [0.4 + 0.5 * j for j in range(5)]
            for j, pos in enumerate(positions):
                if j < current:
                    ax.text(pos, 0.2, "★", fontsize=20, ha="center", va="center", color="black")
                else:
                    ax.text(pos, 0.2, "☆", fontsize=20, ha="center", va="center", color="gray")

            label_y = 0.75
            label = "Clarté des contours" if i == 0 else "Contraste"
            ax.text(2.2, label_y, label, fontsize=9, ha="center", va="top")

    def on_click(self, event):
        for i, ax in enumerate(self.star_axes):
            if event.inaxes == ax:
                xdata = event.xdata
                if xdata is None:
                    return

                positions = [0.4 + 0.5 * j for j in range(5)]
                distances = [abs(xdata - p) for p in positions]
                star_clicked = distances.index(min(distances)) + 1

                key = self.names[self.index]
                if key not in human_scores:
                    human_scores[key] = {'clarte': 0, 'contraste': 0}
                if i == 0:
                    human_scores[key]['clarte'] = star_clicked
                else:
                    human_scores[key]['contraste'] = star_clicked

                self.update_scores_display()
                self.fig.canvas.draw_idle()
                break

    def next(self, event):
        self.index = (self.index + 1) % len(self.images)
        self.update()

    def prev(self, event):
        self.index = (self.index - 1) % len(self.images)
        self.update()

def normalize_to_uint8(image):
    image = np.nan_to_num(image)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    norm_image = (image - min_val) / (max_val - min_val)
    return (norm_image * 255).astype(np.uint8)


# Export des évaluations humaines vers CSV
def export_scores_to_csv(mode="human",scores=[], name_csv="", name_img=[],save_dir=None):
        
    # Chemin du dossier du script actuel
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if mode == "human":
        if name_csv == "": name_csv="human_scores.csv"
        categories = ["Algorithm", "Clarity", "Contrast"]
        values = []
        for algo in human_scores.keys():
            clarte = human_scores[algo].get("clarte", 0)
            contraste = human_scores[algo].get("contraste", 0)
            values.append([algo, clarte, contraste])

    elif mode == "image":
        if name_csv == "": name_csv="image_scores.csv"
        categories = ["nom_image", "score"]
        values = []
        if len(name_img) == 0:
            name_img = [f"Image {i}" for i in range(len(scores))]

        for i in range(len(scores)):
            values.append([name_img[i], scores[i]])
        
    elif mode == "medsam":
        if name_csv == "": name_csv="medsam_scores.csv"
        categories =["Algorithm","score","parameters"]+[f"P_{i}" for i in range(5)]
        values = []
        i=0
        for algo in ALGO_REGISTRY.keys(): 
            
            try :
                score=scores[i]
            except :
                score = -1
            parameters = ALGO_REGISTRY[algo]["params"]
            v=[algo,score,str(list(parameters.keys()))]
            for e in parameters.values():
                v.append(e)
            values.append(v)
            i+=1
    
    elif mode == "human_sequence":
        if name_csv == "":
            name_csv = "human_sequence.csv"
        categories = ["Step", "Algorithm", "Parameters"]
        values = []

        for step, algo, param_vals in scores:
            param_str = ', '.join((repr(v) if type(v)==str else str(v)) for v in param_vals)
            values.append([step, algo, "{" + param_str + "}"])

        filepath = os.path.join(save_dir if 'save_dir' in locals() else script_dir, name_csv)

    # Chemin complet du CSV dans le même dossier
    filepath = os.path.join(save_dir if save_dir else script_dir, name_csv)

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        
        # Title for the columns : ["Algorithm", "Clarity", "Contrast"] if human mode for example
        writer.writerow(categories)
        for v in values:
            writer.writerow(v)  
            
    print(f"[✔] Évaluations exportées vers : {filepath}")
    
class AlgoViewer_medsam:
    
    def __init__(self, images, evals, masks, names=None, score=None):
        self.images = images
        self.evals = evals  # liste de (prob_map, prediction)
        self.masks = masks

        names_with_scores = []
        if names is not None :
            if score is not None:
                for i in range(len(names)):
                    names_with_scores.append(names[i] + f"    -   score = {round(score[i], 3)}")
                self.names = names_with_scores
            else:
                self.names = names
        else:
            
            self.names = [f"Image {i}"+f"    -   score = {round(score[i], 3)}" if score is not None else f"Image {i}" for i in range(len(images))]

        self.index = 0

        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 10))
        plt.subplots_adjust(bottom=0.15)

        self.labels = ["Image transformée", "Probability map", "Prediction", "Mask"]
        self.axes = self.axes.flatten()

        for ax, label in zip(self.axes, self.labels):
            ax.set_title(label)
            ax.axis('off')

        # Boutons navigation
        self.axprev = self.fig.add_axes([0.25, 0.02, 0.2, 0.06])
        self.axnext = self.fig.add_axes([0.55, 0.02, 0.2, 0.06])
        self.bnext = Button(self.axnext, 'Suivant')
        self.bprev = Button(self.axprev, 'Précédent')
        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)

        self.update()

    def update(self):
        img = normalize_to_uint8(self.images[self.index])
        prob_map, prediction = self.evals[self.index]
        prob_map = normalize_to_uint8(prob_map)
        prediction = normalize_to_uint8(prediction)
        mask = self.masks

        images_to_show = [img, prob_map, prediction, mask]
    
        for ax, im, label in zip(self.axes, images_to_show, self.labels):
            ax.clear()

            ax.imshow(im, cmap='gray')
            ax.set_title(label)
            ax.axis('off')

        self.fig.suptitle(self.names[self.index], fontsize=14)
        self.fig.canvas.draw_idle()

    def next(self, event):
        self.index = (self.index + 1) % len(self.images)
        self.update()

    def prev(self, event):
        self.index = (self.index - 1) % len(self.images)
        self.update()


def histograms_pixels(image, mask):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pixels organoïdes
    organoid_pixels = image[mask > 0].flatten()
    # Pixels fond
    background_pixels = image[(mask == 0) & (image > 0)].flatten()

    # Si image est 16 bits, on la normalise en 0-255 pour l'affichage
    if image.max() > 255:
        organoid_pixels = (organoid_pixels / image.max() * 255).astype(np.uint8)
        background_pixels = (background_pixels / image.max() * 255).astype(np.uint8)

    # Création figure avec 2 sous-graphes
    plt.figure(figsize=(8, 3))  # Taille horizontale propre

    bins = np.arange(0, 257)  # 256 bins entre 0-255

    # 🔵 Histogramme organoïdes
    plt.subplot(1, 2, 1)
    plt.hist(organoid_pixels, bins=bins, color='blue', alpha=0.7)
    plt.title('Organoids')
    plt.xlabel('Intensité')
    plt.ylabel('Pixels')
    plt.xlim(0, 255)  # ⬅️ Échelle fixée
    plt.grid(True)

    # 🔴 Histogramme fond
    plt.subplot(1, 2, 2)
    plt.hist(background_pixels, bins=bins, color='red', alpha=0.7)
    plt.title('Background')
    plt.xlabel('Intensity')
    plt.ylabel('Pixels')
    plt.xlim(0, 255)  # ⬅️ Échelle fixée
    plt.grid(True)

    plt.tight_layout()

    # Sauvegarde en mémoire
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    return Image.open(buf)





class TkAlgoEditor:
    def __init__(self, image, mask):
        self.root = tk.Tk()
        self.root.title("Organoid Vision Editor")
        self.root.attributes('-fullscreen', True)

        self.original_image = image.copy()
        self.current_image = image.copy()
        self.mask = mask.copy()

        self.history = [(self.original_image.copy(), "Original Image", {})]
        self.current_index = 0
        self.params = {}
        self.param_entries = {}
        self.saved_folders = {}
        self.bitwise_im1 = None
        self.bitwise_im2 = None

        self.percentile_active=0.05
        self.tresh_active=65
        
        self.selection_radius = 20  # Rayon initial


        self.circle_coords = {"Original Image": None, "Processed Image": None}

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20)

        self.original_label = tk.Label(self.canvas_frame)
        self.original_label.grid(row=0, column=0, padx=5, pady=5)
        self.processed_label = tk.Label(self.canvas_frame)
        self.processed_label.grid(row=0, column=1, padx=5, pady=5)

        self.original_label.bind("<Motion>", self.show_coords_original)
        self.original_label.bind("<Leave>", self.clear_coords)

        self.processed_label.bind("<Motion>", self.show_coords_processed)
        self.processed_label.bind("<Leave>", self.clear_coords)

        self.original_label.bind("<Button-1>", lambda event: self.show_stats(event, self.original_image, "Original Image"))
        self.processed_label.bind("<Button-1>", lambda event: self.show_stats(event, self.current_image, "Processed Image"))

        self.original_label.bind("<MouseWheel>", self.change_radius)
        self.processed_label.bind("<MouseWheel>", self.change_radius)

        self.mask_label = tk.Label(self.control_frame)
        self.mask_label.pack(side=tk.BOTTOM, pady=10)

        self.coord_label = tk.Label(self.canvas_frame, text="", anchor='w')
        self.coord_label.grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=5)

        tk.Label(self.control_frame, text="Select Algorithm:").pack(pady=5)
        self.algo_selector = ttk.Combobox(self.control_frame, values=list(ALGO_REGISTRY.keys()))
        self.current_algo = list(ALGO_REGISTRY.keys())[0]
        self.algo_selector.set(self.current_algo)
        self.algo_selector.pack()
        self.algo_selector.bind("<<ComboboxSelected>>", self.on_algo_change)

        self.param_frame = tk.Frame(self.control_frame)
        self.param_frame.pack(pady=10)

        self.bitwise_buttons_frame = tk.Frame(self.control_frame)
        self.bitwise_buttons_frame.pack(pady=5)
        self.open_im1_btn = tk.Button(self.bitwise_buttons_frame, text="Open im1", command=self.load_im1)
        self.open_im2_btn = tk.Button(self.bitwise_buttons_frame, text="Open im2", command=self.load_im2)


        button_frame1 = tk.Frame(self.control_frame)
        button_frame1.pack(pady=5)

        self.load_image_button = tk.Button(button_frame1, text="Load Image", command=self.load_new_image)
        self.load_image_button.pack(side=tk.LEFT, padx=2)
        self.reset_button = tk.Button(button_frame1, text="Reset", command=self.reset_all)
        self.reset_button.pack(side=tk.LEFT, padx=2)
        
        button_frame2 = tk.Frame(self.control_frame)
        button_frame2.pack(pady=5)

        self.back_button = tk.Button(button_frame2, text="Back", command=self.navigate_back)
        self.back_button.pack(side=tk.LEFT, padx=2)
        self.next_button = tk.Button(button_frame2, text="Next", command=self.navigate_next)
        self.next_button.pack(side=tk.LEFT, padx=2)

        self.apply_button = tk.Button(button_frame2, text="Apply", command=self.apply_current)
        self.apply_button.pack(pady=5)

        button_frame3 = tk.Frame(self.control_frame)
        button_frame3.pack(pady=5)

        self.save_button = tk.Button(button_frame3, text="Save", command=self.save_state)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.load_button = tk.Button(button_frame3, text="Load", command=self.load_state)
        self.load_button.pack(side=tk.LEFT, padx=5)
        

        self.stat_button = tk.Button(self.control_frame, text="Stat", command=self.show_stats_window)
        self.stat_button.pack(pady=5)

        tk.Label(self.control_frame, text="Comment:").pack()
        self.comment_box = tk.Text(self.control_frame, height=4, width=30)
        self.comment_box.pack(pady=10)

        tk.Label(self.control_frame, text="Applied Algorithms:").pack()
        self.history_listbox = tk.Listbox(self.control_frame, height=10, width=40)
        self.history_listbox.pack(pady=10)
        self.history_listbox.bind("<Button-1>", self.open_saved_folder)

                # Gauche
        self.original_label.bind("<Button-1>", lambda e: self.on_click(e, self.original_image, "Original Image"))
        self.processed_label.bind("<Button-1>", lambda e: self.on_click(e, self.processed_image, "Processed Image"))

        # Clic droit (efface)
        self.original_label.bind("<Button-3>", lambda e: self.on_right_click(e, "Original Image"))
        self.processed_label.bind("<Button-3>", lambda e: self.on_right_click(e, "Processed Image"))

        # Molette
        self.original_label.bind("<MouseWheel>", self.change_radius)
        self.processed_label.bind("<MouseWheel>", self.change_radius)

        self.cursor_label = tk.Label(self.root, text="", fg="blue")
        self.cursor_label.pack()

        self.stats_label = tk.Label(self.root, text="", fg="green")
        self.stats_label.pack()

        self.on_algo_change(None)
        self.refresh_history_highlight()
        self.refresh_display()
        self.root.mainloop()

    def load_im1(self):
        path = filedialog.askopenfilename(title="Select Image 1")
        if path: self.bitwise_im1 = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    def load_im2(self):
        path = filedialog.askopenfilename(title="Select Image 2")
        if path: self.bitwise_im2 = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    def on_algo_change(self, event):
        algo = self.algo_selector.get()
        if algo not in ALGO_REGISTRY:
            messagebox.showerror("Error", f"Algorithm '{algo}' not found.")
            return
        self.current_algo = algo
        self.params = ALGO_REGISTRY[algo]["params"].copy()
        self.update_param_fields()

        if algo == "bitwise":
            self.open_im1_btn.pack(side=tk.LEFT, padx=2)
            self.open_im2_btn.pack(side=tk.LEFT, padx=2)
        else:
            self.open_im1_btn.pack_forget()
            self.open_im2_btn.pack_forget()

    def update_param_fields(self):
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_entries.clear()
        for k, v in self.params.items():
            row = tk.Frame(self.param_frame)
            row.pack(fill=tk.X, pady=2)
            label = tk.Label(row, text=k, anchor='w', width=20)
            label.pack(side=tk.LEFT)
            entry = tk.Entry(row)
            entry.insert(0, str(v))
            entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            self.param_entries[k] = entry

    def apply_algo(self):
        func = ALGO_REGISTRY[self.current_algo]["func"]
        try:
            if self.current_algo == "bitwise":
                print("applying bitwise")
                return func(self.bitwise_im1, self.bitwise_im2,**self.params)
            
            return func(self.current_image.copy(), **self.params)
        except Exception as e:
            print(f"[ERROR] {self.current_algo}: {e}")
            return self.current_image

    def apply_current(self):
        try:
            for k, entry in self.param_entries.items():
                try:
                    float(entry.get())
                    self.params[k] = eval(entry.get())
                except:
                    self.params[k] = entry.get()
                
        except Exception as e:
            messagebox.showerror("Parameter Error", str(e))
            return

        new_image = self.apply_algo()
        self.current_image = new_image.copy()

        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
            self.history_listbox.delete(self.current_index + 1, tk.END)

        self.history.append((new_image.copy(), self.current_algo, self.params.copy()))
        self.current_index = len(self.history) - 1

        print(self.params.values())
        values_str = ', '.join( ((str(v)) if type(v)==float else repr(str(v))) for v in self.params.values() )
        self.history_listbox.insert(tk.END, f"{len(self.history)}. {self.current_algo} {{{values_str}}}")
        self.refresh_history_highlight()
        self.refresh_display()

    def navigate_back(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.current_image = self.history[self.current_index][0].copy()
            self.refresh_history_highlight()
            self.refresh_display()

    def navigate_next(self):
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self.current_image = self.history[self.current_index][0].copy()
            self.refresh_history_highlight()
            self.refresh_display()

    def refresh_history_highlight(self):
        for i in range(self.history_listbox.size()):
            bg = "yellow" if i == self.current_index else "white"
            self.history_listbox.itemconfig(i, bg=bg)

    def refresh_display(self):

        def imgtk_from_array(arr, target_size=(512, 512)):
            # Convertir en RGB si besoin
            img = Image.fromarray((arr if arr.ndim == 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)).astype(np.uint8))
            # Redimensionner à taille fixe
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image=img)
        
        self.tk_original = imgtk_from_array(self.original_image,target_size=(512, 512))
        self.tk_processed = imgtk_from_array(self.current_image,target_size=(512, 512))
        self.tk_mask = imgtk_from_array(self.mask,target_size=(256, 256))
        self.original_label.configure(image=self.tk_original)
        self.original_label.image = self.tk_original
        self.processed_label.configure(image=self.tk_processed)
        self.processed_label.image = self.tk_processed
        self.mask_label.configure(image=self.tk_mask)
        self.mask_label.image = self.tk_mask

    def reset_all(self):
        self.current_image = self.original_image.copy()
        self.history = [(self.original_image.copy(), "Original Image", {})]
        self.current_index = 0
        self.history_listbox.delete(0, tk.END)
        self.history_listbox.insert(tk.END, "1. Original Image {}")
        self.refresh_history_highlight()
        self.refresh_display()

    def show_coords_original(self, event):
        x = int(event.x / (self.tk_original.width() / self.original_image.shape[1]))
        y = int(event.y / (self.tk_original.height() / self.original_image.shape[0]))
        if 0 <= x < self.original_image.shape[1] and 0 <= y < self.original_image.shape[0]:
            intensity = self.original_image[y, x]  # (y, x) car numpy est ligne-colonne
            if len(intensity.shape) == 0:  # grayscale
                self.coord_label.config(text=f"Original Image - x: {x}, y: {y}, value: {intensity}")
            else:  # couleur (RGB ou BGR)
                self.coord_label.config(text=f"Original Image - x: {x}, y: {y}, value: {tuple(intensity)}")

    def show_coords_processed(self, event):
        x = int(event.x / (self.tk_processed.width() / self.current_image.shape[1]))
        y = int(event.y / (self.tk_processed.height() / self.current_image.shape[0]))
        if 0 <= x < self.current_image.shape[1] and 0 <= y < self.current_image.shape[0]:
            intensity = self.current_image[y, x]
            if len(intensity.shape) == 0:  # grayscale
                self.coord_label.config(text=f"Processed Image - x: {x}, y: {y}, value: {intensity}")
            else:  # couleur (RGB ou BGR)
                self.coord_label.config(text=f"Processed Image - x: {x}, y: {y}, value: {tuple(intensity)}")

    def on_click(self, event, image, image_type):
        x = int(event.x / (event.widget.winfo_width() / image.shape[1]))
        y = int(event.y / (event.widget.winfo_height() / image.shape[0]))

        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            self.circle_coords[image_type] = (x, y)
            self.show_stats(x, y, image, image_type)
            self.update_image_display(image, image_type)

        if image.shape[1] <= x < 2*image.shape[1] and 0 <= y < image.shape[0]:
            self.circle_coords[image_type] = (x, y)
            self.show_stats(x, y, image, image_type)
            self.update_image_display(image, image_type)
        
    def on_right_click(self, event, image_type):
        self.circle_coords[image_type] = None
        self.coord_label.config(text="")
        self.update_image_display(self.original_image if image_type == "Original Image" else self.processed_image, image_type)

    def clear_coords(self, event):
        self.coord_label.config(text="")

    def on_motion(self, event, image, image_type):
        x = int(event.x / (event.widget.winfo_width() / image.shape[1]))
        y = int(event.y / (event.widget.winfo_height() / image.shape[0]))
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            intensity = image[y, x]
            if np.isscalar(intensity) or len(intensity.shape) == 0:
                text = f"{image_type} - x: {x}, y: {y}, value: {intensity}"
            else:
                text = f"{image_type} - x: {x}, y: {y}, value: {tuple(intensity)}"
            self.cursor_label.config(text=text)
        else:
            self.cursor_label.config(text="")

    def show_stats(self, x, y, image, image_type):
        R = self.selection_radius
        x_min = max(x - R, 0)
        x_max = min(x + R, image.shape[1] - 1)
        y_min = max(y - R, 0)
        y_max = min(y + R, image.shape[0] - 1)

        zone = image[y_min:y_max+1, x_min:x_max+1]

        if zone.ndim == 2 or zone.shape[2] == 1:  # grayscale
            mean = np.mean(zone)
            median = np.median(zone)
            std = np.std(zone)
            stats_text = f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}"
        else:  # couleur
            mean = np.mean(zone, axis=(0, 1))
            median = np.median(zone, axis=(0, 1))
            std = np.std(zone, axis=(0, 1))
            stats_text = (f"Mean: {tuple(np.round(mean,2))}, "
                        f"Median: {tuple(np.round(median,2))}, "
                        f"Std: {tuple(np.round(std,2))}")

        self.stats_label.config(
            text=f"{image_type} - x: {x}, y: {y}\n{stats_text}\nRayon sélection: {R}"
        )
    
    def draw_circle_and_update(self, image, x, y, R, image_type):
        image_with_circle = image.copy()
        cv2.circle(image_with_circle, (x, y), R, (0, 255, 0), 2)  # cercle vert

        image_rgb = cv2.cvtColor(image_with_circle, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        if image_type == "Original Image":
            self.original_tk = img_tk  # stocke pour pas que ça soit garbage collected
            self.original_label.config(image=self.original_tk)
        else:
            self.processed_tk = img_tk
            self.processed_label.config(image=self.processed_tk)


    def change_radius(self, event):
        if event.delta > 0:
            self.selection_radius = min(self.selection_radius + 1, 100)
        else:
            self.selection_radius = max(self.selection_radius - 1, 1)

        self.coord_label.config(text=f"Rayon sélection: {self.selection_radius}")

        # Redessine cercles si déjà cliqués
        for image_type in ["Original Image", "Processed Image"]:
            image = self.original_image if image_type == "Original Image" else self.processed_image
            self.update_image_display(image, image_type)


    def update_image_display(self, image, image_type):

        
        image_with_circle = image.copy()
        coords = self.circle_coords[image_type]
        if coords:
            x, y = coords
            color = (0, 255, 0) if image_type == "Original Image" else (255, 0, 0)
            cv2.circle(image_with_circle, (x, y), self.selection_radius, color, 2)

        image_rgb = cv2.cvtColor(image_with_circle, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)

        # ✅ Fixer une taille constante (par ex 512x512)
        img_pil = img_pil.resize((512, 512), Image.ANTIALIAS)

        img_tk = ImageTk.PhotoImage(img_pil)

        if image_type == "Original Image":
            self.original_tk = img_tk
            self.original_label.config(image=self.original_tk)
        else:
            self.processed_tk = img_tk
            self.processed_label.config(image=self.processed_tk)



    # --- Sauvegarde / Chargement ---

    def save_state(self):
        folder_selected = filedialog.askdirectory(title="Sélectionnez le dossier de destination")
        if not folder_selected:
            return

        # --- Demander un nom de dossier ---
        folder_name = simpledialog.askstring("Nom du dossier", "Entrez un nom pour le dossier de sauvegarde :")
        if not folder_name:
            return

        save_folder = os.path.join(folder_selected, folder_name)
        os.makedirs(save_folder, exist_ok=True)

        # --- Sauvegarder l’image traitée uniquement ---
        #cv2.imwrite(os.path.join(save_folder, "processed.png"), self.current_image)

        # --- Sauvegarder la séquence d’algorithmes ---
        history_algos = []
        for i, (img, algo, params) in enumerate(self.history):
            #save toutes les images
            cv2.imwrite(os.path.join(save_folder, f"{i}.png"), img)
            
            param_vals = list(params.values())
            history_algos.append((i + 1, algo, param_vals))
        export_scores_to_csv(mode="human_sequence", scores=history_algos,
                            name_csv="applied_algorithms.csv", name_img=None,
                            save_dir=save_folder)

        # --- Sauvegarder toutes les stats (sauf 'text') ---
        try:
            stats = pixel_distribution(self.current_image, self.mask, percentile=self.percentile_active, thresh=self.tresh_active)
            with open(os.path.join(save_folder, "stats.csv"), "w", newline="") as f:
                writer = csv.writer(f, delimiter=";")

                # Extraire toutes les clés sauf 'text'
                keys = [k for k in stats["organoid"].keys() if k != "text"]
                writer.writerow(["Zone"] + keys)

                for zone in ["organoid", "background"]:
                    row = [zone] + [stats[zone].get(k, "") for k in keys]
                    writer.writerow(row)
        except Exception as e:
            print(f"[WARNING] Stat export failed: {e}")

        # --- Interface ---
        listbox_text = f"Saved: {folder_name}"
        idx = self.history_listbox.size()
        self.history_listbox.insert(tk.END, listbox_text)
        self.saved_folders[idx] = save_folder
        messagebox.showinfo("Succès", f"Images et stats sauvegardées dans : {folder_name}")


    def load_new_image(self):
        path = filedialog.askopenfilename(
            title="Select New Original Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tif *.bmp")]
        )
        if not path:
            return

        # Lire la nouvelle image en N&B ou couleur
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
                
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mettre à jour les attributs
        self.original_image = img.copy()
        self.current_image = img.copy()

        # Reset historique
        self.history = [(self.original_image.copy(), "Original Image", {})]
        self.current_index = 0
        self.history_listbox.delete(0, tk.END)
        self.history_listbox.insert(tk.END, "1. Original Image {}")

        # Refresh display
        self.refresh_history_highlight()
        self.refresh_display()

    def load_state(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return
        try:
            # Reset l'image courante à l'image d'origine déjà chargée
            self.current_image = self.original_image.copy()
            self.history = [(self.original_image.copy(), "Original Image", {})]
            self.history_listbox.delete(0, tk.END)
            self.history_listbox.insert(tk.END, "1. Original Image {}")

            with open(file_path, mode="r", newline="") as f:
                reader = csv.reader(f, delimiter=';')
                next(reader)  # skip header
                for row in reader:
                    if len(row) < 3:
                        continue
                    step, algo, param_str = row
                    param_str = param_str.strip("{} ")
                    print(f"param_str: {param_str}")
                    param_vals = [eval(val.strip()) for val in param_str.split(',') if val.strip()]
                    
                    if algo not in ALGO_REGISTRY:
                        print(f"[WARN] Algorithme {algo} inconnu. Ignoré.")
                        continue
                    # Reconstruire le dict des paramètres à partir des valeurs
                    
                    param_names = list(ALGO_REGISTRY[algo]["params"].keys())
                    
                    params = dict(zip(param_names, param_vals))
                    self.current_algo = algo
                    self.params = params

                    img=cv2.imread(os.path.join(os.path.dirname(file_path), f"{int(step)-1}.png"), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise ValueError(f"Image for step {step} not found or failed to load.")
                    self.current_image = img
                    self.history.append((img, algo, params.copy()))
                    values_str = ', '.join(map(str, param_vals))
                    self.history_listbox.insert(tk.END, f"{len(self.history)}. {algo} {{{values_str}}}")

            self.current_index = len(self.history) - 1
            self.refresh_history_highlight()
            self.refresh_display()
            messagebox.showinfo("Success", "Sequence loaded from CSV file.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Erreur lors du chargement : {e}")

    def reset_all(self):
        self.current_image = self.original_image.copy()
        self.history = [(self.original_image.copy(), "Original Image", {})]
        self.current_index = 0
        self.history_listbox.delete(0, tk.END)
        self.history_listbox.insert(tk.END, "1. Original Image {}")
        self.refresh_history_highlight()
        self.refresh_display()


    # --- Statistiques ---
    def show_stats_window(self):
        def update_stats():
            try:
                # Lire le percentile depuis l'input
                new_percentile = float(percentile_entry.get())
                new_tresh = float(threshold_entry.get())
                self.percentile_active= new_percentile
                self.tresh_active = new_tresh

                if not 0 < new_percentile < 1 or not 0 < new_tresh < 255:
                    raise ValueError

                # Appeler à nouveau la fonction pixel_distribution avec le nouveau percentile
                new_stats = pixel_distribution(self.current_image, self.mask, new_percentile, thresh=new_tresh)

                # Mettre à jour les labels
                organoid_label.config(text=f"Organoid Stats:\n{new_stats['organoid']['text']}")
                background_label.config(text=f"Background Stats:\n{new_stats['background']['text']}")

            except ValueError:
                tk.messagebox.showerror("Erreur", "0<percentile<1 and 0<= treshold<=255.")

        # 1. Générer histogramme + stats (valeurs par défaut percentile=0.1)
        hist_img = histograms_pixels(self.current_image, self.mask)
        stats = pixel_distribution(self.current_image, self.mask, percentile=self.percentile_active, thresh=self.tresh_active)

        # ✅ Resize propre
        max_width = 600
        w_percent = (max_width / float(hist_img.width))
        h_size = int((float(hist_img.height) * float(w_percent)))
        hist_img = hist_img.resize((max_width, h_size), Image.Resampling.LANCZOS)
        histogram_photo = ImageTk.PhotoImage(hist_img)

        # 2. Crée nouvelle fenêtre
        stat_window = tk.Toplevel(self.root)
        stat_window.title("Histogram & Stats")

        # 3. Affiche l'histogramme
        img_label = tk.Label(stat_window, image=histogram_photo)
        img_label.image = histogram_photo
        img_label.pack()

        # ✅ 4. Ajoute une frame pour le contrôle percentile + bouton
        control_frame = tk.Frame(stat_window)
        control_frame.pack(pady=10)

        tk.Label(control_frame, text="Percentile (0-1) :").pack(side=tk.LEFT, padx=5)
        percentile_entry = tk.Entry(control_frame, width=5)
        
        percentile_entry.insert(0, self.percentile_active)
        percentile_entry.pack(side=tk.LEFT, padx=5)

        # ✅ 4.1 Ajoute une frame pour le contrôle threshold + bouton

        tk.Label(control_frame, text="Threshold (0-255) :").pack(side=tk.LEFT, padx=5)
        threshold_entry = tk.Entry(control_frame, width=5)
        
        threshold_entry.insert(0, self.tresh_active)
        threshold_entry.pack(side=tk.LEFT, padx=5)
        
        apply_button = tk.Button(control_frame, text="Apply", command=update_stats)
        apply_button.pack(side=tk.LEFT, padx=5)

        # 5. Cadre pour le texte stats
        text_frame = tk.Frame(stat_window)
        text_frame.pack(fill=tk.X, pady=10)

        organoid_label = tk.Label(text_frame, text=f"Organoid Stats:\n{stats['organoid']['text']}", anchor='w', justify='left')
        organoid_label.pack(side=tk.LEFT, padx=20)

        background_label = tk.Label(text_frame, text=f"Background Stats:\n{stats['background']['text']}", anchor='e', justify='left')
        background_label.pack(side=tk.RIGHT, padx=20)


        


    # --- Affichage des images ---
        def imgtk_from_array(arr, scale=2.0):
            img = Image.fromarray((arr if arr.ndim == 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)).astype(np.uint8))
            w, h = img.size
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            return ImageTk.PhotoImage(image=img)

        self.tk_original = imgtk_from_array(self.original_image)
        self.tk_processed = imgtk_from_array(self.current_image)
        self.tk_mask = imgtk_from_array(self.mask, scale=1.0)

        self.original_label.configure(image=self.tk_original)
        self.original_label.image = self.tk_original
        self.processed_label.configure(image=self.tk_processed)
        self.processed_label.image = self.tk_processed
        self.mask_label.configure(image=self.tk_mask)
        self.mask_label.image = self.tk_mask

    def open_saved_folder(self, event):
        idx = self.history_listbox.curselection()[0]
        if idx in self.saved_folders:
            folder = self.saved_folders[idx]
            system = platform.system()
            try:
                if system == "Windows":
                    os.startfile(folder)
                elif system == "Darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d’ouvrir le dossier : {e}")


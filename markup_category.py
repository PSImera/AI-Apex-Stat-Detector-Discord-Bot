import tkinter as tk
from tkinter import filedialog
import os
import csv
from PIL import Image, ImageTk


class ImageClassifierApp:
    ''' Application for markup classification csv file for image dataset'''
    def __init__(self, root, image_folder, categories, output_csv, annotated_csv):
        self.root = root
        self.image_folder = image_folder
        self.categories = categories
        self.output_csv = output_csv
        
        # Read already annotated files from the given CSV
        self.annotated_files = set()
        if os.path.exists(annotated_csv):
            with open(annotated_csv, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                self.annotated_files = {row[0] for row in reader}

        # Get list of unannotated image files
        self.image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')) and f not in self.annotated_files
        ]

        # Index to start marking from
        self.current_index = 0

        # Load existing classifications if CSV exists
        self.classifications = {}
        if os.path.exists(output_csv):
            with open(output_csv, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                self.classifications = {rows[0]: rows[1] for rows in reader}

        # GUI setup
        self.image_label = tk.Label(root)
        self.image_label.pack()
        
        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack()

        for index, category in enumerate(categories):
            btn = tk.Button(self.buttons_frame, text=f"{index + 1}: {category}", 
                            command=lambda c=category: self.save_classification(c))
            btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.status_label = tk.Label(root, text="")
        self.status_label.pack()

        # Bind keyboard events
        self.root.bind("<Key>", self.on_key_press)

        self.show_image()

    def show_image(self):
        if self.current_index >= len(self.image_files):
            self.status_label.config(text="Все изображения размечены!")
            self.image_label.config(image='')
            return

        image_path = os.path.join(self.image_folder, self.image_files[self.current_index])
        img = Image.open(image_path)
        img.thumbnail((1280, 720))  # Resize for display
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_image)
        self.status_label.config(text=f"Изображение {self.current_index + 1} из {len(self.image_files)}")

    def save_classification(self, category):
        file_name = self.image_files[self.current_index]
        self.classifications[file_name] = category

        with open(self.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for file, cat in self.classifications.items():
                writer.writerow([file, cat])

        self.current_index += 1
        self.show_image()

    def on_key_press(self, event):
        try:
            # Convert key to an integer and map it to category
            key_num = int(event.char)  # Get number key pressed
            if 1 <= key_num <= len(self.categories):  # Ensure it's within range
                category = self.categories[key_num - 1]
                self.save_classification(category)
        except ValueError:
            # Ignore non-numeric key presses
            pass

# Main
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Классификатор изображений")

    # image_folder = filedialog.askdirectory(title="Выберите папку с изображениями")
    # output_csv = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Выберите CSV для сохранения результатов")
    # annotated_csv = filedialog.askopenfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Выберите CSV с уже размеченными файлами")

    image_folder = 'datasets/ap'
    output_csv = 'datasets/ap_target.csv'

    categories = [1, 2, 3, 4, 5, 6, -666]

    app = ImageClassifierApp(root, image_folder, categories, output_csv, output_csv)
    root.mainloop()
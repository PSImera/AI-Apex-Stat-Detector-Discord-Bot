import tkinter as tk
from tkinter import filedialog
import os
import csv
from PIL import Image, ImageTk

class TextAnnotatorApp:
    ''' Application for markup text annotation csv file for image dataset'''
    def __init__(self, root, image_folder, output_csv):
        self.root = root
        self.image_folder = image_folder
        self.output_csv = output_csv
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
        self.current_index = 0

        # Load existing annotations if CSV exists
        self.annotations = {}
        if os.path.exists(output_csv):
            with open(output_csv, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header row
                self.annotations = {rows[0]: rows[1] for rows in reader}

        # GUI setup
        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.entry_frame = tk.Frame(root)
        self.entry_frame.pack()

        self.entry_label = tk.Label(self.entry_frame, text="Введите текстовую разметку:")
        self.entry_label.pack(side=tk.LEFT)

        self.text_entry = tk.Entry(self.entry_frame, width=50)
        self.text_entry.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(root, text="Сохранить", command=self.save_annotation)
        self.save_button.pack(pady=5)

        self.status_label = tk.Label(root, text="")
        self.status_label.pack()

        self.show_image()

    def show_image(self):
        if self.current_index >= len(self.image_files):
            self.status_label.config(text="Все изображения размечены!")
            self.image_label.config(image='')
            self.text_entry.delete(0, tk.END)
            self.text_entry.config(state=tk.DISABLED)
            return

        image_path = os.path.join(self.image_folder, self.image_files[self.current_index])
        img = Image.open(image_path)
        img.thumbnail((800, 600))  # Resize for display
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_image)

        file_name = self.image_files[self.current_index]
        self.text_entry.delete(0, tk.END)
        if file_name in self.annotations:
            self.text_entry.insert(0, self.annotations[file_name])

        self.status_label.config(text=f"Изображение {self.current_index + 1} из {len(self.image_files)}")

    def save_annotation(self):
        file_name = self.image_files[self.current_index]
        annotation_text = self.text_entry.get()
        self.annotations[file_name] = annotation_text

        with open(self.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "words"])
            for file, text in self.annotations.items():
                writer.writerow([file, text])

        self.current_index += 1
        self.show_image()

# Main
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Текстовый аннотатор изображений")

    image_folder = filedialog.askdirectory(title="Выберите папку с изображениями")
    output_csv = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Выберите CSV для сохранения результатов")

    app = TextAnnotatorApp(root, image_folder, output_csv)
    root.mainloop()

    
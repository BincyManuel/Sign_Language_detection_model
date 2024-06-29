import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np


model=load_model("sign_language_detection_model.keras")

class SignLanguageDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection")
        self.root.geometry("500x500")

        self.uploaded_image = None
        self.image_label = tk.Label(self.root, text="upload an image")
        self.image_label.pack()
        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()
        self.detect_button = tk.Button(self.root, text="Detect Sign", command=self.detect_sign, state=tk.DISABLED)
        self.detect_button.pack()
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

    def upload_image(self):
        filepath = filedialog.askopenfilename()
        self.uploaded_image = cv2.imread(filepath)
        self.image_label.config(text="Image uploaded successfully")
        self.detect_button.config(state=tk.NORMAL)

    def detect_sign(self):
        resized_image = cv2.resize(self.uploaded_image, (224, 224))
        resized_image = resized_image / 255.0
        resized_image = np.expand_dims(resized_image, axis=0)

        prediction = model.predict(resized_image)[0]
        predicted_class = np.argmax(prediction)

        sign_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        predicted_sign = sign_classes[predicted_class]

        self.result_label.config(text=f"Detected sign: {predicted_sign}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = SignLanguageDetectionGUI(root)
    root.mainloop()
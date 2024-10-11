import cv2
import pytesseract
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

class LicensePlateRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition")

        self.original_image_label = tk.Label(self.root)
        self.gray_image_label = tk.Label(self.root)
        self.edges_image_label = tk.Label(self.root)
        self.plate_number_label = tk.Label(self.root, text="Recognized License Plate Number: ")

        self.original_image_label.grid(row=0, column=0, padx=5, pady=5)
        self.gray_image_label.grid(row=0, column=1, padx=5, pady=5)
        self.edges_image_label.grid(row=0, column=2, padx=5, pady=5)
        self.plate_number_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        self.load_image_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_image_button.grid(row=2, column=0, columnspan=3, pady=10)

        self.image = None
        self.gray = None
        self.edges = None
        self.license_plate_image = None
        self.plate_number = ""

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if path:
            self.process_image(path)

    def process_image(self, path):
        self.image = cv2.imread(path)
        if self.image is None:
            messagebox.showerror("Error", "Unable to load image.")
            return

        # Convert to grayscale
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(self.gray, (5, 5), 0)

        # Edge detection
        self.edges = cv2.Canny(blur, 50, 150)

        # Convert images to RGB format for displaying in Tkinter
        original_image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        gray_image_rgb = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2RGB)
        edges_image_rgb = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2RGB)

        # Convert images to PIL format
        original_image_pil = Image.fromarray(original_image_rgb)
        gray_image_pil = Image.fromarray(gray_image_rgb)
        edges_image_pil = Image.fromarray(edges_image_rgb)

        # Convert PIL images to Tkinter format
        self.original_image_tk = ImageTk.PhotoImage(original_image_pil)
        self.gray_image_tk = ImageTk.PhotoImage(gray_image_pil)
        self.edges_image_tk = ImageTk.PhotoImage(edges_image_pil)

        # Update labels with images
        self.original_image_label.config(image=self.original_image_tk)
        self.gray_image_label.config(image=self.gray_image_tk)
        self.edges_image_label.config(image=self.edges_image_tk)

        # Find contours
        contours, _ = cv2.findContours(self.edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours based on area, keeping only the largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # Loop through contours to find the license plate
        license_plate = None
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # The license plate should be a rectangle with 4 corners
            if len(approx) == 4:
                license_plate = approx
                break

        if license_plate is not None:
            # Create a mask for the license plate
            mask = np.zeros(self.gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [license_plate], -1, 255, -1)

            # Bitwise-and to extract the license plate area
            self.license_plate_image = cv2.bitwise_and(self.gray, self.gray, mask=mask)

            # Crop the license plate area
            x, y, w, h = cv2.boundingRect(license_plate)
            self.license_plate_image = self.license_plate_image[y:y + h, x:x + w]

            # Apply thresholding to preprocess for OCR
            _, self.license_plate_image = cv2.threshold(self.license_plate_image, 150, 255, cv2.THRESH_BINARY)

            # Configure tesseract to recognize only alphanumeric characters
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

            # Perform OCR on the license plate image
            self.plate_number = pytesseract.image_to_string(self.license_plate_image, config=custom_config).strip()

            # Update plate number label
            self.plate_number_label.config(text=f"Recognized License Plate Number: {self.plate_number}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognizer(root)
    app.run()
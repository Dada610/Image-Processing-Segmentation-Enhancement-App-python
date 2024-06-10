import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, ttk , messagebox
import cv2 
from PIL import Image, ImageTk 
import numpy as np
from ttkbootstrap import Style



class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Processing Application")
        ctk.set_appearance_mode("dark")  
        self.style = Style(theme='darkly')

        self.top_frame = ctk.CTkFrame(master)
        self.top_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.top_frame.columnconfigure([0, 1, 2], weight=1)

        self.bottom_frame = ctk.CTkFrame(master)
        self.bottom_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.bottom_frame.columnconfigure([0, 1, 2, 3], weight=1)

        self.original_label = ctk.CTkLabel(self.top_frame, text="Original Image")
        self.original_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.enhanced_label = ctk.CTkLabel(self.top_frame, text="Enhanced Image")
        self.enhanced_label.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        self.segmented_label = ctk.CTkLabel(self.top_frame, text="Segmented Image")
        self.segmented_label.grid(row=0, column=2, padx=10, pady=5, sticky="nsew")

        self.original_image_label = ttk.Label(self.top_frame)
        self.original_image_label.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.enhanced_image_label = ttk.Label(self.top_frame)
        self.enhanced_image_label.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.segmented_image_label = ttk.Label(self.top_frame)
        self.segmented_image_label.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")

        self.load_button = ctk.CTkButton(self.bottom_frame, text="Open Image", command=self.load_image, fg_color="#4CAF50")  # green
        self.load_button.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.enhance_button = ctk.CTkButton(self.bottom_frame, text="Enhance", command=self.enhance_image, fg_color="#FFC107")  # amber
        self.enhance_button.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.num_entry = ctk.CTkEntry(self.bottom_frame,placeholder_text="Enter  gamma")  # amber
        self.num_entry.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")

        self.segment_button = ctk.CTkButton(self.bottom_frame, text="Segment", command=self.segment_image, fg_color="#F44336")  # red
        self.segment_button.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        self.final_screen_button = ctk.CTkButton(self.bottom_frame, text="Final Screen", command=self.display_final_screen, fg_color="#007BFF")  # blue
        self.final_screen_button.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")

        self.ZoomIn_button = ctk.CTkButton(self.bottom_frame, text="Zoom In", command=self.Zoom_In, fg_color="#17A2B8")  # teal
        self.ZoomIn_button.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.ZoomOut_button = ctk.CTkButton(self.bottom_frame, text="Zoom Out", command=self.Zoom_Out, fg_color="#6C757D")  # gray
        self.ZoomOut_button.grid(row=1, column=3, padx=5, pady=5, sticky="nsew")

        combobox_style_name = 'Custom.TCombobox'
        self.style.map(combobox_style_name, fieldbackground=[('readonly', '#285577')], selectbackground=[('readonly', '#5085A5')])
        self.style.configure(combobox_style_name, arrowsize=10, fieldbackground="#285577", background="#285577", foreground="#FFFFFF", selectbackground="#5085A5", selectforeground="#FFFFFF")

        self.enhancement_technique = tk.StringVar()
        self.enhancement_dropdown = ctk.CTkComboBox(self.bottom_frame,
                                                    values=["Negative Transformation", "Thresholding",
                                                            "Log Transformation", "Power-Law Transformation",
                                                            "Piecewise Linear Transformation", "Gray Level Slicing",
                                                            "Bit Plane Slicing"],
                                                    variable=self.enhancement_technique)
        self.enhancement_dropdown.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.segmentation_technique = tk.StringVar()
        self.segmentation_dropdown = ctk.CTkComboBox(self.bottom_frame,
                                                     values=["Thresholding", "Canny Edge Detection",
                                                             "Region Growing Segmentation"],
                                                     variable=self.segmentation_technique)
        self.segmentation_dropdown.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
        self.image = None
        self.processed_images = []

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image, self.original_image_label)

    def Zoom_In(self):
        if self.image is not None:
            zoom_factor = 1.2
            new_width = int(self.image.shape[1] * zoom_factor)
            new_height = int(self.image.shape[0] * zoom_factor)
            self.image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            self.display_image(self.image, self.original_image_label)

    def Zoom_Out(self):
        if self.image is not None:
            min_zoom_factor = max(0.1, self.image.shape[0] / (self.image.shape[0] * 1.2))
            zoom_factor = min(1.2, min_zoom_factor)
            new_width = int(self.image.shape[1] * zoom_factor)
            new_height = int(self.image.shape[0] * zoom_factor)
            self.image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            self.display_image(self.image, self.original_image_label)

    def enhance_image(self):
        if self.image is not None:
            selected_enhancement = self.enhancement_technique.get()
            if selected_enhancement == "Negative Transformation":
                enhanced_image = cv2.bitwise_not(self.image)
                self.processed_images.append(enhanced_image)
                self.display_image(enhanced_image, self.enhanced_image_label)
            elif selected_enhancement == "Thresholding":
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                _, enhanced_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
                self.processed_images.append(enhanced_image)
                self.display_image(enhanced_image, self.enhanced_image_label)
            elif selected_enhancement == "Log Transformation":
                enhanced_image = self.log_transformation(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
                self.processed_images.append(enhanced_image)
                self.display_image(enhanced_image, self.enhanced_image_label)
            elif selected_enhancement == "Power-Law Transformation":
                gamma = float(self.num_entry.get())
                enhanced_image = self.power_law_transformation(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), gamma)
                self.processed_images.append(enhanced_image)
                self.display_image(enhanced_image, self.enhanced_image_label)
            elif selected_enhancement == "Piecewise Linear Transformation":
                enhanced_image = self.piecewise_linear_transformation(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
                self.processed_images.append(enhanced_image)
                self.display_image(enhanced_image, self.enhanced_image_label)
            elif selected_enhancement == "Gray Level Slicing":
                enhanced_image = self.gray_level_slicing(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
                self.processed_images.append(enhanced_image)
                self.display_image(enhanced_image, self.enhanced_image_label)
            elif selected_enhancement == "Bit Plane Slicing":
                enhanced_image = self.bit_plane_slicing(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
                self.processed_images.append(enhanced_image)
                self.display_image(enhanced_image, self.enhanced_image_label)
            else:
                messagebox.showwarning("Warning", "Please select an enhancement technique.")

    def segment_image(self):
        if self.image is not None:
            selected_segmentation = self.segmentation_technique.get()
            if selected_segmentation == "Thresholding":
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                _, segmented_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
                self.processed_images.append(segmented_image)
                self.display_image(segmented_image, self.segmented_image_label)
            elif selected_segmentation == "Canny Edge Detection":
                edges = cv2.Canny(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), 100, 200)
                self.processed_images.append(edges)
                self.display_image(edges, self.segmented_image_label)
            elif selected_segmentation == "Watershed Segmentation":
                segmented_image = self.watershed_segmentation()
                self.processed_images.append(segmented_image)
                self.display_image(segmented_image, self.segmented_image_label)
            elif selected_segmentation == "Region Growing Segmentation":
                segmented_image = self.region_growing_segmentation()
                self.processed_images.append(segmented_image)
                self.display_image(segmented_image, self.segmented_image_label)
            
            else:
                messagebox.showwarning("Warning", "Please select a segmentation technique.")

    def display_image(self, image, label_widget):
        print("Displaying image...")
        print("Image shape:", image.shape if image is not None else "None")

        max_width = 300
        max_height = 800

        if image is None:
            print("Error: Image is None")
            return

        if image.shape[0] <= 0 or image.shape[1] <= 0:
            print("Error: Invalid image dimensions")
            return

        if image.shape[1] > max_width or image.shape[0] > max_height:
            aspect_ratio = image.shape[1] / image.shape[0]

            if aspect_ratio > 1: 
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:  
                new_height = max_height
                new_width = int(max_height * aspect_ratio)

            print("New width:", new_width)
            print("New height:", new_height)

            try:
                resized_image = cv2.resize(image, (new_width, new_height))
            except Exception as e:
                print("Error resizing image:", e)
                return
        else:
            resized_image = image

        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = Image.fromarray(resized_image)
        resized_image = ImageTk.PhotoImage(resized_image)

        label_widget.configure(image=resized_image)
        label_widget.image = resized_image

    def log_transformation(self, image_gray):
        c = 255 / np.log(1 + np.max(image_gray)) 
        log_transformed = c * (np.log(image_gray + 1))
        return np.array(log_transformed, dtype=np.uint8)  

    def power_law_transformation(self, image_gray, gamma):
        power_law = np.power(image_gray / float(np.max(image_gray)), gamma)
        return np.array(255 * power_law, dtype=np.uint8)

    def piecewise_linear_transformation(self, image_gray):
        min_val = np.min(image_gray)
        max_val = np.max(image_gray)
        alpha = 255.0 / (max_val - min_val)
        beta = -min_val * alpha
        return np.array(alpha * image_gray + beta, dtype=np.uint8)

    def gray_level_slicing(self, image_gray):
        lower_threshold = 100
        upper_threshold = 200
        gray_level_sliced = np.copy(image_gray)
        mask = (gray_level_sliced >= lower_threshold) & (gray_level_sliced <= upper_threshold)
        gray_level_sliced[mask] = 255
        return gray_level_sliced

    def bit_plane_slicing(self, image_gray):
        bit_planes = [2 ** i for i in range(8)]
        bit_planes_reconstructed = np.zeros_like(image_gray)
        for i, plane in enumerate(bit_planes):
            bit_planes_reconstructed += ((image_gray >> i) & 1) * plane
        return bit_planes_reconstructed

    

    def region_growing_segmentation(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            seed_point = (0, 0) 
            segmented_image = self.region_growing(gray_image, seed_point)
            self.processed_images.append(segmented_image)
            self.display_image(segmented_image, self.segmented_image_label)
            return segmented_image

    def region_growing(self, image, seed_point, threshold=10):
        segmented_image = np.zeros_like(image, dtype=np.uint8)
        queue = [seed_point]
        while queue:
            x, y = queue.pop(0)
            if segmented_image[x, y] == 0:
                segmented_image[x, y] = 255
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if 0 <= x + i < image.shape[0] and 0 <= y + j < image.shape[1]:
                            if abs(int(image[x, y]) - int(image[x + i, y + j])) < threshold:
                                queue.append((x + i, y + j))
        return segmented_image

    
        

    def display_final_screen(self):
        if self.image is not None and self.processed_images:
            final_screen = tk.Toplevel(self.master)
            final_screen.title("Final Screen")

            original_label = ttk.Label(final_screen, text="Original Image")
            original_label.grid(row=0, column=0, padx=5, pady=5)
            self.display_image(self.image, original_label)

            processed_images_resized = []
            for processed_image in self.processed_images:
                processed_image_resized = cv2.resize(processed_image, (self.image.shape[1], self.image.shape[0]))
                processed_images_resized.append(processed_image_resized)

            for i, processed_image in enumerate(processed_images_resized):
                label = ttk.Label(final_screen, text=f"Processed Image {i + 1}")
                label.grid(row=0, column=i + 1, padx=5, pady=5)
                self.display_image(processed_image, label)


def main():
    root = ctk.CTk()
    app = ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


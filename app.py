from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog as fd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from class_lib.paint import ImagePainter

class GUI(ctk.CTk):
    def __init__(self, painter: 'ImagePainter'):
        super().__init__()
        self.title("Image Colorizer")
        self.iconphoto(False, ImageTk.PhotoImage(file="icon.ico"))
        self.painter = painter
        image_size = self.painter.image_size
        self.geometry(f"{2*image_size[0]+50}x{image_size[1]+100}")
        self.minsize(width=2*image_size[0]+50, height=image_size[1]+100)
        
        self.select_button = ctk.CTkButton(self, width=150, height=35, corner_radius=6, border_width=2,
                                            text= "Select Image", command=self.select_image)
        self.color_button = ctk.CTkButton(self, width=150, height=35, corner_radius=6, border_width=2,
                                            text= "Color it!", command=self.color_image, state="disabled")
        self.save_button = ctk.CTkButton(self, width=150, height=35, corner_radius=6, border_width=2,
                                            text= "Save Image", command=self.save_image, state="disabled")
        self.image1_box = ctk.CTkLabel(self, width=image_size[0], height=image_size[1], text="Original Image")
        self.image2_box = ctk.CTkLabel(self, width=image_size[0], height=image_size[1], text="Colored Image")
        
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.grid_columnconfigure((0, 1), weight= 1)
        self.select_button.grid(row=0, column=0, pady=4)
        self.image1_box.grid(row=1, column=0, pady=4, padx=4)
        self.image2_box.grid(row=1, column=1, pady=4, padx=4)
        self.color_button.grid(row=2, column=0, pady=4)
        self.save_button.grid(row=2, column=1, pady=4)
      
    def select_image(self):
        imagepath = fd.askopenfilename(title="Select an image", 
                                       filetypes= (("JPEG", ["*.jpg", "*.jfif"]), 
                                                   ("PNG", "*.png"), 
                                                   ("All files", "*.*")))
        if imagepath:
            image_size = self.painter.image_size
            
            self.image1 = Image.open(imagepath).convert('RGB')
            im1 = ctk.CTkImage(self.image1, size=image_size)
            self.image1_box.configure(text="", image=im1)
            self.image2_box.destroy()
            self.image2_box = ctk.CTkLabel(self, width=image_size[0], height=image_size[1], text="Colored Image")
            self.image2_box.grid(row=1, column=1, pady=4, padx=4)
            self.color_button.configure(state="normal")
            self.save_button.configure(state="disabled")
        
    def color_image(self):
        self.image2_box.configure(text = "Coloring Image\nPlease wait...")
        self.image2_box.update()
        self.image2 = self.painter.paint(self.image1)
        im2 = ctk.CTkImage(self.image2, size=self.painter.image_size)
        self.image2_box.configure(text="", image=im2)
        self.save_button.configure(state="normal")
        
    def save_image(self):
        savepath = fd.asksaveasfilename(initialfile="untitled.jpg", defaultextension=".jpg",
                                        filetypes= (("JPEG", "*.jpg"), 
                                                    ("PNG", "*.png"), 
                                                    ("All files", "*.*")))
        if savepath:
            self.image2.save(savepath)
        
    
if __name__ == "__main__":
    from class_lib.paint import ImagePainter
    from settings import *
    
    GUI(ImagePainter(MODEL_PATH, IMAGE_SIZE)).mainloop()
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

# Funções de filtros
def apply_filter_laplace(imagem, sensitivity):
    laplacian = cv2.Laplacian(imagem, cv2.CV_64F, ksize=sensitivity)
    return cv2.convertScaleAbs(laplacian)

def apply_filter_sobel(imagem, sensitivity):
    sobel_x = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=sensitivity)
    sobel_y = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=sensitivity)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    return cv2.convertScaleAbs(sobel)

def apply_filter_gaussian(imagem, kernel_size):
    return cv2.GaussianBlur(imagem, (kernel_size, kernel_size), 5)

def apply_filter_average(imagem, kernel_size):
    return cv2.blur(imagem, (kernel_size, kernel_size))

# Operações morfológicas
def apply_morphological_operation(imagem, operation, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == "dilate":
        return cv2.dilate(imagem, kernel, iterations=1)
    elif operation == "erode":
        return cv2.erode(imagem, kernel, iterations=1)
    elif operation == "open":
        return cv2.morphologyEx(imagem, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        return cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, kernel)

# Limiarização (Threshold)
def apply_threshold(imagem, method, thresh_value=127):
    if method == "binary":
        _, thresh = cv2.threshold(imagem, thresh_value, 255, cv2.THRESH_BINARY)
    elif method == "otsu":
        _, thresh = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Aplicar filtros
def apply_filter(filter_type):
    global edited_img
    if original_img is None:  # Garantir que uma imagem foi carregada
        return
    
    # Sempre começar a partir da imagem original
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    if filter_type == "low_pass_gaussian":
        kernel_size = gaussian_slider.get()
        if kernel_size % 2 == 0:
            kernel_size += 1
        filtered_img = apply_filter_gaussian(gray_img, kernel_size)
        show_slider(gaussian_slider)
    elif filter_type == "low_pass_average":
        kernel_size = average_slider.get()
        if kernel_size % 2 == 0:
            kernel_size += 1
        filtered_img = apply_filter_average(gray_img, kernel_size)
        show_slider(average_slider)
    elif filter_type == "high_pass_laplace":
        sensitivity = laplace_slider.get()
        if sensitivity % 2 == 0:
            sensitivity += 1
        filtered_img = apply_filter_laplace(gray_img, sensitivity)
        show_slider(laplace_slider)
    elif filter_type == "high_pass_sobel":
        sensitivity = sobel_slider.get()
        if sensitivity % 2 == 0:
            sensitivity += 1
        filtered_img = apply_filter_sobel(gray_img, sensitivity)
        show_slider(sobel_slider)
    elif filter_type in ["morph_dilate", "morph_erode", "morph_open", "morph_close"]:
        kernel_size = morph_slider.get()
        if kernel_size % 2 == 0:
            kernel_size += 1
        filtered_img = apply_morphological_operation(gray_img, filter_type.split("_")[1], kernel_size)
        show_slider(morph_slider)
    elif filter_type == "threshold_binary":
        thresh_value = binary_threshold_slider.get()
        filtered_img = apply_threshold(gray_img, "binary", thresh_value)
        show_slider(binary_threshold_slider)
    elif filter_type == "threshold_otsu":
        filtered_img = apply_threshold(gray_img, "otsu")
        hide_all_sliders()

    # Atualizar a imagem editada com a versão filtrada
    edited_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    display_image(edited_img, original=False)
def preview_filter(filter_type):
    def update_preview(val=None):
        apply_filter(filter_type)
    return update_preview

def show_slider(slider):
    sliders = [gaussian_slider, average_slider, laplace_slider, sobel_slider, morph_slider, binary_threshold_slider]
    for s in sliders:
        if s == slider:
            s.pack(side="top", fill="x", padx=10)
        else:
            s.pack_forget()

def hide_all_sliders():
    gaussian_slider.pack_forget()
    average_slider.pack_forget()
    laplace_slider.pack_forget()
    sobel_slider.pack_forget()
    morph_slider.pack_forget()
    binary_threshold_slider.pack_forget()

# Funções de imagem
def load_image():
    global img_cv, edited_img, original_img
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = cv2.imread(file_path)
        original_img = img_cv.copy()
        edited_img = img_cv.copy()
        display_image(img_cv, original=True)
        refresh_canvas()

def reset_image():
    global edited_img
    if original_img is not None:
        edited_img = original_img.copy()
        display_image(edited_img, original=False)

def display_image(img, original=False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_width, img_height = img_pil.size
    max_size = 500
    img_pil.thumbnail((max_size, max_size))
    img_tk = ImageTk.PhotoImage(img_pil)
    canvas_width, canvas_height = max_size, max_size
    x_offset = (canvas_width - img_pil.width) // 2
    y_offset = (canvas_height - img_pil.height) // 2

    if original:
        original_image_canvas.delete("all")
        original_image_canvas.image = img_tk
        original_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
    else:
        edited_image_canvas.delete("all")
        edited_image_canvas.image = img_tk
        edited_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)

def refresh_canvas():
    edited_image_canvas.delete("all")

# Janela principal
root = tk.Tk()
root.title("Image Processing App")
root.geometry("1085x800")
root.config(bg="#2e2e2e")

# Frame principal com scrollbar
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(main_frame, bg="#2e2e2e")
scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

canvas.configure(yscrollcommand=scrollbar.set)

# Inner frame
inner_frame = tk.Frame(canvas, bg="#2e2e2e")
canvas.create_window((0, 0), window=inner_frame, anchor="nw")

# Atualizar região de rolagem
def update_scrollregion(event=None):
    canvas.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

inner_frame.bind("<Configure>", update_scrollregion)

# Menu de Arquivo
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Canvases
original_image_canvas = tk.Canvas(inner_frame, width=500, height=500, bg="white")
original_image_canvas.grid(row=0, column=0, padx=10, pady=10)

edited_image_canvas = tk.Canvas(inner_frame, width=500, height=500, bg="white")
edited_image_canvas.grid(row=0, column=1, padx=10, pady=10)

reset_button = tk.Button(inner_frame, text="Reset Image", command=reset_image, bg="#6c757d", fg="white")
reset_button.grid(row=1, column=1, pady=10)

# Barra de ferramentas para os botões
button_frame = tk.Frame(inner_frame, bg="#343a40", padx=10, pady=10)
button_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

# Filtros Low Pass
low_pass_label = tk.Label(button_frame, text="Low Pass Filters:", bg="#343a40", fg="white")
low_pass_label.grid(row=0, column=0, padx=5)

btn_avg = tk.Button(button_frame, text="Average", command=lambda: apply_filter("low_pass_average"), width=12, bg="#007bff", fg="white")
btn_avg.grid(row=0, column=1, padx=5)

btn_gaussian = tk.Button(button_frame, text="Gaussian", command=lambda: apply_filter("low_pass_gaussian"), width=12, bg="#007bff", fg="white")
btn_gaussian.grid(row=0, column=2, padx=5)

# Filtros High Pass
high_pass_label = tk.Label(button_frame, text="High Pass Filters:", bg="#343a40", fg="white")
high_pass_label.grid(row=1, column=0, padx=5)

btn_laplace = tk.Button(button_frame, text="Laplace", command=lambda: apply_filter("high_pass_laplace"), width=12, bg="#28a745", fg="white")
btn_laplace.grid(row=1, column=1, padx=5)

btn_sobel = tk.Button(button_frame, text="Sobel", command=lambda: apply_filter("high_pass_sobel"), width=12, bg="#28a745", fg="white")
btn_sobel.grid(row=1, column=2, padx=5)

# Operações Morfológicas
morph_label = tk.Label(button_frame, text="Morphological Operations:", bg="#343a40", fg="white")
morph_label.grid(row=2, column=0, padx=5)

btn_dilate = tk.Button(button_frame, text="Dilate", command=lambda: apply_filter("morph_dilate"), width=12, bg="#ffc107", fg="white")
btn_dilate.grid(row=2, column=1, padx=5)

btn_erode = tk.Button(button_frame, text="Erode", command=lambda: apply_filter("morph_erode"), width=12, bg="#ffc107", fg="white")
btn_erode.grid(row=2, column=2, padx=5)

btn_open = tk.Button(button_frame, text="Open", command=lambda: apply_filter("morph_open"), width=12, bg="#ffc107", fg="white")
btn_open.grid(row=3, column=1, padx=5)

btn_close = tk.Button(button_frame, text="Close", command=lambda: apply_filter("morph_close"), width=12, bg="#ffc107", fg="white")
btn_close.grid(row=3, column=2, padx=5)

# Threshold
threshold_label = tk.Label(button_frame, text="Thresholds:", bg="#343a40", fg="white")
threshold_label.grid(row=4, column=0, padx=5)

btn_threshold_binary = tk.Button(button_frame, text="Binary", command=lambda: apply_filter("threshold_binary"), width=12, bg="#6f42c1", fg="white")
btn_threshold_binary.grid(row=4, column=1, padx=5)

btn_threshold_otsu = tk.Button(button_frame, text="Otsu", command=lambda: apply_filter("threshold_otsu"), width=12, bg="#6f42c1", fg="white")
btn_threshold_otsu.grid(row=4, column=2, padx=5)

# Sliders para filtros
slider_frame = tk.Frame(inner_frame, bg="#343a40")
slider_frame.grid(row=3, column=0, columnspan=2, pady=10)

gaussian_slider = tk.Scale(slider_frame, from_=1, to=31, orient=tk.HORIZONTAL, label="Gaussian Kernel Size", length=400, bg="#343a40", fg="white", command=preview_filter("low_pass_gaussian"))
gaussian_slider.set(15)

average_slider = tk.Scale(slider_frame, from_=1, to=31, orient=tk.HORIZONTAL, label="Average Kernel Size", length=400, bg="#343a40", fg="white", command=preview_filter("low_pass_average"))
average_slider.set(7)

laplace_slider = tk.Scale(slider_frame, from_=1, to=31, orient=tk.HORIZONTAL, label="Laplace Sensitivity", length=400, bg="#343a40", fg="white", command=preview_filter("high_pass_laplace"))
laplace_slider.set(3)

sobel_slider = tk.Scale(slider_frame, from_=1, to=31, orient=tk.HORIZONTAL, label="Sobel Sensitivity", length=400, bg="#343a40", fg="white", command=preview_filter("high_pass_sobel"))
sobel_slider.set(3)

morph_slider = tk.Scale(slider_frame, from_=1, to=31, orient=tk.HORIZONTAL, label="Morphological Kernel Size", length=400, bg="#343a40", fg="white", command=preview_filter("morph_dilate"))
morph_slider.set(5)

binary_threshold_slider = tk.Scale(slider_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Binary Threshold", length=400, bg="#343a40", fg="white", command=preview_filter("threshold_binary"))
binary_threshold_slider.set(127)

root.mainloop()
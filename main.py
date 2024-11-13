import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

def apply_filter_laplace(imagem, ganho=1.0): 
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    altura, largura = imagem.shape
    imagem_saida = np.zeros((altura, largura), dtype=np.float32)

    for i in range(1, altura-1):
        for j in range(1, largura-1):
            vizinhança = imagem[i-1:i+2, j-1:j+2]
            valor_filtrado = np.sum(kernel * vizinhança) * ganho
            imagem_saida[i, j] = np.clip(valor_filtrado, 0, 255)

    return cv2.normalize(imagem_saida, None, 0, 255, cv2.NORM_MINMAX)

def apply_filter_sobel(imagem):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0,  0,  0], [1,  2,  1]])
    altura, largura = imagem.shape
    imagem_saida_x = np.zeros((altura, largura), dtype=np.float32)
    imagem_saida_y = np.zeros((altura, largura), dtype=np.float32)

    for i in range(1, altura-1):
        for j in range(1, largura-1):
            vizinhança = imagem[i-1:i+2, j-1:j+2]
            imagem_saida_x[i, j] = np.sum(sobel_x * vizinhança)
            imagem_saida_y[i, j] = np.sum(sobel_y * vizinhança)

    imagem_saida = np.sqrt(imagem_saida_x**2 + imagem_saida_y**2)
    return cv2.normalize(imagem_saida, None, 0, 255, cv2.NORM_MINMAX)

def apply_filter_gaussian(imagem, tamanho=15, sigma=5):
    kernel = kernel_gaussiano(tamanho=tamanho, sigma=sigma)
    altura, largura = imagem.shape
    imagem_saida = np.zeros((altura, largura), dtype=np.float32)

    offset = tamanho // 2
    for i in range(offset, altura - offset):
        for j in range(offset, largura - offset):
            vizinhança = imagem[i - offset:i + offset + 1, j - offset:j + offset + 1]
            valor_filtrado = np.sum(kernel * vizinhança)
            imagem_saida[i, j] = valor_filtrado

    # Normaliza a imagem de saída
    imagem_saida = cv2.normalize(imagem_saida, None, 0, 255, cv2.NORM_MINMAX)
    return imagem_saida



def kernel_gaussiano(tamanho=7, sigma=1.5):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                     np.exp(-(((x - (tamanho - 1) / 2) ** 2 + (y - (tamanho - 1) / 2) ** 2) / (2 * sigma ** 2))),
        (tamanho, tamanho)
    )
    return kernel / np.sum(kernel)


def apply_filter_average(imagem):
    kernel = np.ones((7, 7)) / 49.0
    altura, largura = imagem.shape
    imagem_saida = np.zeros((altura, largura), dtype=np.float32)

    for i in range(3, altura-3):
        for j in range(3, largura-3):
            vizinhança = imagem[i-3:i+4, j-3:j+4]
            valor_filtrado = np.sum(kernel * vizinhança)
            imagem_saida[i, j] = valor_filtrado

    imagem_saida = cv2.normalize(imagem_saida, None, 0, 255, cv2.NORM_MINMAX)
    return imagem_saida

def apply_morphological_operation(imagem, operation, kernel_size=5):
    imagem_binaria = np.where(imagem >= 127, 255, 0).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    pad = kernel_size // 2
    padded_image = np.pad(imagem_binaria, pad, mode='constant', constant_values=0)

    # Funções auxiliares
    def erode(img):
        output = np.zeros_like(img, dtype=np.uint8)
        for i in range(pad, img.shape[0] + pad):
            for j in range(pad, img.shape[1] + pad):
                region = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
                if np.all(region == 255):
                    output[i - pad, j - pad] = 255
                else:
                    output[i - pad, j - pad] = 0
        return output

    def dilate(img):
        output = np.zeros_like(img, dtype=np.uint8)
        for i in range(pad, img.shape[0] + pad):
            for j in range(pad, img.shape[1] + pad):
                region = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
                if np.any(region == 255):
                    output[i - pad, j - pad] = 255
                else:
                    output[i - pad, j - pad] = 0
        return output

    def opening(img):
        # Abertura: erosão seguida de dilatação
        eroded_img = erode(img)
        padded_eroded_img = np.pad(eroded_img, pad, mode='constant', constant_values=0)
        return dilate(padded_eroded_img)

    def closing(img):
        # Fechamento: dilatação seguida de erosão
        dilated_img = dilate(img)
        padded_dilated_img = np.pad(dilated_img, pad, mode='constant', constant_values=0)
        return erode(padded_dilated_img)

    # Executa a operação selecionada
    if operation == "erosion":
        return erode(imagem_binaria)
    elif operation == "dilation":
        return dilate(imagem_binaria)
    elif operation == "opening":
        return opening(imagem_binaria)
    elif operation == "closing":
        return closing(imagem_binaria)
    else:
        raise ValueError("Operação desconhecida. Escolha entre 'erosion', 'dilation', 'opening' ou 'closing'.")

def apply_thresholding(imagem, method="binary", threshold_value=127, adjustment_factor=1.0):
    if method == "binary":
        thresholded = binary_threshold(imagem, threshold_value)
        
    elif method == "otsu":
        threshold_value = otsu_threshold(imagem, adjustment_factor)
        thresholded = binary_threshold(imagem, threshold_value)
            
    return thresholded



def binary_threshold(imagem, threshold_value):

    thresholded = np.zeros_like(imagem, dtype=np.uint8)
    thresholded[imagem > threshold_value] = 255
    return thresholded

def otsu_threshold(imagem, adjustment_factor=1.0):
    histogram, _ = np.histogram(imagem, bins=256, range=(0, 256))
    total_pixels = imagem.size
    sum_total = np.sum(np.arange(256) * histogram)
    
    sum_background, weight_background, weight_foreground = 0, 0, 0
    max_variance, threshold_value = 0, 0
    
    for i in range(256):
        weight_background += histogram[i]
        if weight_background == 0:
            continue
            
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
            
        sum_background += i * histogram[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance_between > max_variance:
            max_variance = variance_between
            threshold_value = i
    
    adjusted_threshold = int(threshold_value * adjustment_factor)
    adjusted_threshold = np.clip(adjusted_threshold, 0, 255)
    
    return adjusted_threshold

def apply_filter(filter_type):
    if img_cv is None:
        return
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    if filter_type == "low_pass_gaussian":
        filtered_img = apply_filter_gaussian(gray_img, tamanho=15, sigma=5.0)
    elif filter_type == "low_pass_average":
        filtered_img = apply_filter_average(gray_img)
    elif filter_type == "high_pass_laplace":
        filtered_img = apply_filter_laplace(gray_img, ganho=4.0)
    elif filter_type == "high_pass_sobel":
        filtered_img = apply_filter_sobel(gray_img)
    elif filter_type == "morph_erosion":
        filtered_img = apply_morphological_operation(gray_img, "erosion")
    elif filter_type == "morph_dilation":
        filtered_img = apply_morphological_operation(gray_img, "dilation")
    elif filter_type == "morph_opening":
        filtered_img = apply_morphological_operation(gray_img, "opening")
    elif filter_type == "morph_closing":
        filtered_img = apply_morphological_operation(gray_img, "closing")
    elif filter_type == "threshold_binary":
        threshold_value = 79
        filtered_img = apply_thresholding(gray_img, "binary", threshold_value)
    elif filter_type == "otsu":
        filtered_img = apply_thresholding(gray_img, "otsu", adjustment_factor=0.5)
    
    display_image(cv2.cvtColor(np.uint8(filtered_img), cv2.COLOR_GRAY2BGR), original=False)

def load_image():
    global img_cv
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = cv2.imread(file_path)
        display_image(img_cv, original=True)
        refresh_canvas()

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

root = tk.Tk()
root.title("Image Processing App")
root.geometry("1085x600")
root.config(bg="#2e2e2e")

img_cv = None

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filters", menu=filters_menu)

low_pass_menu = tk.Menu(filters_menu, tearoff=0)
low_pass_menu.add_command(label="Average", command=lambda: apply_filter("low_pass_average"))
low_pass_menu.add_command(label="Gaussian", command=lambda: apply_filter("low_pass_gaussian"))

high_pass_menu = tk.Menu(filters_menu, tearoff=0)
high_pass_menu.add_command(label="Laplace", command=lambda: apply_filter("high_pass_laplace"))
high_pass_menu.add_command(label="Sobel", command=lambda: apply_filter("high_pass_sobel"))

morph_menu = tk.Menu(filters_menu, tearoff=0)
morph_menu.add_command(label="Erosion", command=lambda: apply_filter("morph_erosion"))
morph_menu.add_command(label="Dilation", command=lambda: apply_filter("morph_dilation"))
morph_menu.add_command(label="Opening", command=lambda: apply_filter("morph_opening"))
morph_menu.add_command(label="Closing", command=lambda: apply_filter("morph_closing"))

threshold_menu = tk.Menu(filters_menu, tearoff=0)
threshold_menu.add_command(label="Binary Threshold", command=lambda: apply_filter("threshold_binary"))
threshold_menu.add_command(label="otsu", command=lambda: apply_filter("otsu"))

filters_menu.add_cascade(label="Low Pass Filter", menu=low_pass_menu)
filters_menu.add_cascade(label="High Pass Filter", menu=high_pass_menu)
filters_menu.add_cascade(label="Morphological Operations", menu=morph_menu)
filters_menu.add_cascade(label="Thresholding", menu=threshold_menu)

original_image_canvas = tk.Canvas(root, width=500, height=500, bg="white")
original_image_canvas.grid(row=0, column=0, padx=10, pady=10)

edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="white")
edited_image_canvas.grid(row=0, column=1, padx=10, pady=10)

root.mainloop()

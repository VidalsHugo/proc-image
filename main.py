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
    kernel = cv2.getGaussianKernel(tamanho, sigma)
    return cv2.filter2D(imagem, -1, kernel @ kernel.T)

def apply_filter_average(imagem):
    kernel = np.ones((7, 7)) / 49.0
    return cv2.filter2D(imagem, -1, kernel)

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
    
    display_image(cv2.cvtColor(np.uint8(filtered_img), cv2.COLOR_GRAY2BGR), original=False)


def load_image():
    global img_cv
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = cv2.imread(file_path)
        display_image(img_cv, original=True)  # Exibe a imagem original
        refresh_canvas()

def display_image(img, original=False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Obtém o tamanho da imagem orifinal
    img_width, img_height = img_pil.size
    
    # Redimensional a imagem para caber no canvas se for muito grande
    max_size = 500
    img_pil.thumbnail((max_size, max_size))  # Maintain aspect ratio
    img_tk = ImageTk.PhotoImage(img_pil)

    # Calcula a posição para centralizar a imagem dentro do canvas se for menor
    canvas_width, canvas_height = max_size, max_size
    x_offset = (canvas_width - img_pil.width) // 2
    y_offset = (canvas_height - img_pil.height) // 2

    if original:
        original_image_canvas.delete("all")  # Limpa a canvas
        original_image_canvas.image = img_tk  # Mantém a referência viva - garbage collection
        original_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
    else:
        edited_image_canvas.delete("all")  # Limapa a canvas
        edited_image_canvas.image = img_tk
        edited_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)


def refresh_canvas():
    edited_image_canvas.delete("all")  # Limpa a canvas para exibir a nova imagem

# Definindo a GUI
root = tk.Tk()
root.title("Image Processing App")

# Define o tamanho da janela da aplicação 1200x800
root.geometry("1085x550")

# Define a cor de fundo da janela
root.config(bg="#2e2e2e")

img_cv = None

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# File menu (sem alteração)
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Filtros com submenus
filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filters", menu=filters_menu)

# Low Pass Filter submenu
low_pass_menu = tk.Menu(filters_menu, tearoff=0)
low_pass_menu.add_command(label="Average", command=lambda: apply_filter("low_pass_average"))
low_pass_menu.add_command(label="Gaussian", command=lambda: apply_filter("low_pass_gaussian"))

# High Pass Filter submenu
high_pass_menu = tk.Menu(filters_menu, tearoff=0)
high_pass_menu.add_command(label="Laplace", command=lambda: apply_filter("high_pass_laplace"))
high_pass_menu.add_command(label="Sobel", command=lambda: apply_filter("high_pass_sobel"))

# Adiciona submenus ao menu principal de filtros
filters_menu.add_cascade(label="Low Pass Filter", menu=low_pass_menu)
filters_menu.add_cascade(label="High Pass Filter", menu=high_pass_menu)

# Criação das canvas para as imagens (original e editada)
original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=20, pady=20)

root.mainloop()


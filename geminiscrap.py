import pandas as pd
import PIL
import openpyxl
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
import math
from PIL import ImageFilter,ImageDraw
import ast
import os
import google.generativeai as genai
from google import generativeai
from google.generativeai import types
from PIL import Image
import json
import requests

apikey='AIzaSyBX3MViIYToWGa-L34VxArB8-1iwF-EOXo'

inputimg='C:\\Users\\juana\\Documents\\FlyCowOCR\\img-input2'



e1=[]
e2=[]
e3=[]
e4=[]
e5=[]
e6=[]
e7=[]
e8=[]
e9=[]
e10=[]
rowless_used=[]


def scale_width_and_pad_down(image, target_width=680, target_height=675, color=(255, 255, 255)):
    original_width, original_height = image.size
    if original_width != target_width:
        scale_factor = target_width / original_width
        new_height = int(original_height * scale_factor)
        image = image.resize((target_width, new_height), Image.LANCZOS)
    new_width, new_height = image.size
    if new_height < target_height:
        new_image = Image.new("RGB", (target_width, target_height), color)
        new_image.paste(image, (0, 0))
        return new_image
    else:
        return image

def find_column_boundaries(image, threshold=0.95, min_gap_width=5):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = img_array[:, :, 0]

    height, width = img_array.shape
    white_columns = []

    for x in range(width):
        col = img_array[:, x]
        white_ratio = np.mean(col > 240)
        if white_ratio >= threshold:
            white_columns.append(x)

    boundaries = []
    start = None
    for i in range(1, len(white_columns)):
        if white_columns[i] != white_columns[i - 1] + 1:
            if start is not None and white_columns[i - 1] - start >= min_gap_width:
                boundaries.append((start, white_columns[i - 1]))
            start = white_columns[i]
        elif start is None:
            start = white_columns[i - 1]

    if start is not None and white_columns[-1] - start >= min_gap_width:
        boundaries.append((start, white_columns[-1]))

    column_cuts = []
    prev_cut = 0
    for start, end in boundaries:
        cut = (prev_cut, start)
        if cut[1] - cut[0] > 10:
            column_cuts.append(cut)
        prev_cut = end
    column_cuts.append((prev_cut, width))
    return column_cuts
def cortar_por_primera_linea_negra(img, umbral_negro=75, tolerancia_blancos_ratio=0.31, debug=True):

    img_np = np.array(img)

    height, width = img_np.shape
    tolerancia_blancos = int(width * tolerancia_blancos_ratio)

    for y in range(height):
        blancos = np.sum(img_np[y] > umbral_negro)
        if blancos <= tolerancia_blancos:
            if debug:
                print(f"black line search in y = {y} with {blancos} white pixels.")
            cortada = img.crop((0, y, width, height))
            return cortada

    
    return img

import time
from google.api_core.exceptions import ResourceExhausted
genai.configure(api_key=apikey)
# ...

for filename in os.listdir(inputimg):
    filepath = os.path.join(inputimg, filename)
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    print(f"Procesando: {filename}")
    openimg = Image.open(filepath).convert('RGB')
    openimg = scale_width_and_pad_down(openimg)
    openimg = openimg.convert('L')
    openimg = openimg.crop((0, 59, openimg.width, openimg.height))

    enhancer = ImageEnhance.Contrast(openimg)
    openimg = openimg.filter(ImageFilter.GaussianBlur(radius=.40))
    openimg = enhancer.enhance(2)
    openimg = cortar_por_primera_linea_negra(openimg)
    imgtpocr = openimg

    column_boundaries = find_column_boundaries(openimg)
    row_size = 13
    row_less = 0
    pivotcolum = imgtpocr.crop((column_boundaries[5][0]-5, 0, column_boundaries[5][1]+15, openimg.height))

    for x in range(1):
        width, height = pivotcolum.size
        n_round = math.floor(height / (row_size - 1.5))  
        for e in range(n_round):
            top = e * row_size
            bottom = top + row_size
            row_pivotcolum = pivotcolum.crop((0, top + row_less, width, bottom + row_less))
            while True:
                try:
                    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
                    response = model.generate_content(
                        [row_pivotcolum, "Extract all the text or numbers from this image, only respond with what is extracted as it is in the image"]
                    )
                    break
                except ResourceExhausted:
                    time.sleep(40)
            keyword = response.text.replace('\n', ' ').strip()
            if keyword == 'No Ice':
                    
                print('keyword search in row less ' + str(row_less), keyword)
                rowless_used.append(row_less)
                for i, (x0, x1) in enumerate(column_boundaries):
                    if x0 + 10 < x1 and i == 1:
                        col_crop = imgtpocr.crop((x0 - 38, 0, x1 + 15, openimg.height))
                    elif x0 + 10 < x1 and i == 0:
                        col_crop = imgtpocr.crop((x0 - x0, 0, x1 + 30, openimg.height))
                    elif x0 + 10 < x1:
                        col_crop = imgtpocr.crop((x0 - 5, 0, x1 + 15, openimg.height))
                    width, height = col_crop.size
                    row_crop = col_crop.crop((0, top, width, bottom+23)) 

                    while True:
                        try:
                            model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
                            response = model.generate_content(
                                [row_crop, "Extract all the text or numbers from this image, only respond with what is extracted as it is in the image"]
                            )
                            break
                        except ResourceExhausted:
                            time.sleep(40)

                    text = response.text.replace('\n', ',')
                    if i == 0:
                        e1.append(text)
                    elif i == 1:
                        e2.append(text)
                    elif i == 2:
                        e3.append(text)
                    elif i == 3:
                        e4.append(text)
                    elif i == 4:
                        e5.append(text)
                    elif i == 5:
                        e6.append(text)
                    elif i == 6:
                        e7.append(text)
                    elif i == 7:
                        e8.append(text)
                    elif i == 8:
                        e9.append(text)

                    print('Spin ' + str(x), 'column ' + str(i), 'page ' + str(filename), 'row less: ' + str(row_less))
        row_less += 3


sample=({
  'column1':e1,
  'column2':e2,
  'column3':e3,
  'column4':e4,
  'column5':e5,
  'column6':e6,
  'column7':e7,
  'column8':e8,
  'column9':e9,
})


pre_data=pd.DataFrame(sample)


pre_data.drop_duplicates(inplace=True)


pre_data.reset_index(inplace=True)

df=pre_data.index
df = pd.DataFrame(df, columns=['QTY'])  




df['Description']=pre_data['column1']
df['Note/Carrier']=None
df['Note/Carrier']=df['Description'].str.extract(r'\(([a-zA-Z]+)\)')
df['Face or Leg'] = pre_data['column2'].apply(
    lambda x: x.split(' ')[0] if isinstance(x, str) and x.strip() else None
)

df['Offset Type'] = pre_data['column2'].apply(
    lambda x: ' '.join(x.split(' ')[1:]) if isinstance(x, str) and x.strip() else None
)

def get_part(x, idx):
    if isinstance(x, str):
        parts = x.split(',')
        if len(parts) > idx:
            return parts[idx].strip()
    return None

df['Horizontal'] = pre_data['column3'].apply(lambda x: get_part(x, 0))
df['Lateral']    = pre_data['column3'].apply(lambda x: get_part(x, 1))
df['Vertical']   = pre_data['column3'].apply(lambda x: get_part(x, 2))
df['Azimuth Adjustment(degree)']= pre_data['column4']
df['Placement(ft)']=pre_data['column5']
def safe_split(x, index):
    if isinstance(x, str):
        parts = x.split(',')
        if len(parts) > index:
            return parts[index].strip()
    return None

df['CaAa Front (No Ice) ft^2']     = pre_data['column7'].apply(lambda x: safe_split(x, 0))
df['CaAa Front (1/2" Ice) ft^2']   = pre_data['column7'].apply(lambda x: safe_split(x, 1))
df['CaAa Side (No Ice) ft^2']=pre_data['column8'].apply(lambda x: safe_split(x, 0))
df['CaAa Side (1/2" Ice) ft^2']=pre_data['column8'].apply(lambda x: safe_split(x, 1))
df['Weight (No Ice) Weight']=pre_data['column9'].apply(lambda x: safe_split(x, 0))
df['Weight (1/2" Ice) Weight']=pre_data['column9'].apply(lambda x: safe_split(x, 1))



df.to_csv('document_on_text.csv',index=False)
df.to_excel('document_on_text.xlsx',index=False)
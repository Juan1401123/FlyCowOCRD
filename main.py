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


pytesseract.pytesseract.tesseract_cmd =r'C:\Program Files\Tesseract-OCR\tesseract.exe'


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
                print(f"Línea negra encontrada en y = {y} con {blancos} píxeles blancos.")
            cortada = img.crop((0, y, width, height))
            return cortada

    
    return img

listpivot=['Noe,1/2"Ice,1"Ice','No Ice,1/2"Ice,1"Ice','NoIe,1/2"Ice,1"Ice','NoIce,1/2"Ice,1"Ice','1"Ice,Noe,/2"Ice','1"Ice,Noe,1/2"Ice','1"Ice,Noe,1/"Ice','1"Ice,NoIce,1/2"Ice','1"Ice,NoIce,/2"Ice']
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
    openimg = openimg.point(lambda p: 0 if p < 100 else 255)
    openimg = enhancer.enhance(2)
    openimg=cortar_por_primera_linea_negra(openimg)
    # openimg = openimg.filter(ImageFilter.GaussianBlur(radius=.01))
    openimg=openimg.filter(ImageFilter.MedianFilter(size=1))
    imgtpocr=openimg



    column_boundaries = find_column_boundaries(openimg)
    row_size=38
    row_less=-3
    pivotcolum=imgtpocr.crop((column_boundaries[5][0]-5, 0, column_boundaries[5][1]+15, openimg.height))
    for x in range(7):
         width, height = pivotcolum.size
         n_round=math.floor(height/row_size-2)  
         for e in range(n_round):
             top = e * row_size
             bottom = top + row_size
             row_pivotcolum = pivotcolum.crop((0, top+row_less, width, bottom+row_less))
             keyword=pytesseract.image_to_string(row_pivotcolum.resize((row_pivotcolum.width * 8, row_pivotcolum.height * 8), Image.LANCZOS), config = r'--psm 6 --oem 3 -l eng -c tessedit_char_whitelist=No1/2"eIc,').strip().replace('\n',',') 
             if keyword in listpivot:
                  for i, (x0, x1) in enumerate(column_boundaries):
                    if x0+10 < x1 and i==1:
                     col_crop = imgtpocr.crop((x0-38, 0, x1+15, openimg.height))
                    elif x0+10 < x1 and i==0:
                      col_crop = imgtpocr.crop((x0-x0, 0, x1+30, openimg.height))
                    elif x0+10 < x1:
                     col_crop = imgtpocr.crop((x0-5, 0, x1+15, openimg.height))
                    width, height = col_crop.size
                    for e in range(n_round):
                       row_crop = col_crop.crop((0, top+row_less, width, bottom+row_less)) 
                       if i==0:
                          element1 = pytesseract.image_to_string(row_crop.resize((row_crop.width * 5, row_crop.height * 5), Image.LANCZOS), config=r'--psm 6 --oem 3 -l eng').strip().replace('\n',',')
                          e1.append(element1)
                       elif i==1:
                           element2 = pytesseract.image_to_string(row_crop.resize((row_crop.width * 5, row_crop.height * 5), Image.LANCZOS), config=r'--psm 6 --oem 3 -l eng').strip()
                           e2.append(element2)   
                       elif i==2:
                           element3 = pytesseract.image_to_string(row_crop.resize((row_crop.width * 5, row_crop.height * 5), Image.LANCZOS), config=r'--psm 6 --oem 3 -l eng -c tessedit_char_whitelist=0123456789.').strip().replace('\n',',')  
                           e3.append(element3)   
                       elif i==3:
                           element4 = pytesseract.image_to_string(row_crop.resize((row_crop.width * 5, row_crop.height * 5), Image.LANCZOS), config=r'--psm 7 --oem 3 -l eng -c tessedit_char_whitelist=0123456789.').strip().replace('\n',',') 
                           e4.append(element4)      
                       elif i==4:
                           element5 = pytesseract.image_to_string(row_crop.resize((row_crop.width * 5, row_crop.height * 5), Image.LANCZOS), config=r'--psm 7 --oem 3 -l eng -c tessedit_char_whitelist=0123456789.').strip().replace('\n',',') 
                           e5.append(element5)      
                       elif i==5:
                           element6 = pytesseract.image_to_string(row_crop.resize((row_crop.width * 8, row_crop.height * 8), Image.LANCZOS), config = r'--psm 6 --oem 3 -l eng -c tessedit_char_whitelist=No1/2"eIc,').strip().replace('\n',',')    
                           e6.append(element6)   
                       elif i==6:
                           element7 = pytesseract.image_to_string(row_crop.resize((row_crop.width * 5, row_crop.height * 5), Image.LANCZOS), config=r'--psm 6 --oem 3 -l eng -c tessedit_char_whitelist=0123456789.').strip().replace('\n',',') 
                           e7.append(element7)      
                       elif i==7: 
                           element8 = pytesseract.image_to_string(row_crop.resize((row_crop.width * 8, row_crop.height * 8), Image.LANCZOS), config=r'--psm 6 --oem 3 -l eng -c tessedit_char_whitelist=0123456789.').strip().replace('\n',',') 
                           e8.append(element8)   
                       elif i==8:           
                           element9 = pytesseract.image_to_string(row_crop.resize((row_crop.width * 8, row_crop.height * 8), Image.LANCZOS), config=r'--psm 6 --oem 3 -l eng -c tessedit_char_whitelist=0123456789.').strip().replace('\n',',') 
                           e9.append(element9) 
                       print(x,i)
         row_less+=3  
    





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
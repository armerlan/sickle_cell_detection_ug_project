import os
import pandas as pd
import cv2 as cv

def load_images(path):
    df = {'image': [], 'label': []}
    for subdir, _, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                label = 1 if 'positive' in subdir else 0
                img_path = os.path.join(subdir, file)
                img = cv.imread(img_path)
                if img is not None:
                    # Resize the image if necessary (224x224 in this case)
                    img = cv.resize(img, (224, 224))
                    df['image'].append(img)
                    df['label'].append(label)
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(df)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

'''
directory = 'sickle-cell-img'
df = load_images(directory)
print(df.head())'''
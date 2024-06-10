import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Input
# from tensorflow.random import set_seed
# from tensorflow.keras.backend import clear_session
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os

def readdata(samples_per_class):
  X = []
  y = []

  ds_path = 'dataset'
  folders = os.listdir(ds_path)
  for folder in folders:
    files = os.listdir(os.path.join(ds_path, folder))
    for f in files[:samples_per_class]:
      if f.endswith('.png'):
        img = Image.open(os.path.join(ds_path, folder, f))
        img = np.array(img)
        X.append(img)
        y.append(folder)
  
  return np.array(X), np.array(y)

st.title('Handwriting Alphabet Training')

tabs = st.tabs(["Model Training", "Drawable Canvas"])

with tabs[0]:
  
  st.header("Model Training")
  
  samples_per_class = st.slider('Samples per Class', 100, 10000, 1000)
  
  if st.button('Load dataset'):
    X , y = readdata(samples_per_class)
    st.write(y.size)
  
  
  if st.toggle('View Data'):
    fig, axs = plt.subplots(10, 10)
    fig.set_figheight(6)
    fig.set_figwidth(6)
    for i in range(26):
      for j in range(10):
        target = np.random.choice(np.where((y == i))[0])
        axs[i][j].axis('off')
        axs[i][j].imshow(X[target].reshape(28,28), cmap='gray')
    
    st.pyplot(fig)

  epochs = st.slider('Number of Epochs', 1, 20, 2)
  test_size = st.slider('Test Size', 0.1, 0.5, 0.1)
	
  # if st.button('Train Model'):
    
  #   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
  #   y_train_ohe = to_categorical(y_train, num_classes=26)
  #   y_test_ohe = to_categorical(y_test, num_classes=26)
    
  #   model = Sequential()
  #   model.add(Input(shape=X_train.shape[1:]))
  #   model.add(Dense(26, activation='softmax'))
  #   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
  #   model.summary()
    
  #   history = model.fit(X_train, y_train_ohe, epochs = epochs, verbose=1)
        
  #   st.write(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
  #   st.write(f"Test Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
  #   st.line_chart(history.history['accuracy'], use_container_width=True)
  #   st.line_chart(history.history['val_accuracy'], use_container_width=True)
  #   st.line_chart(history.history['loss'], use_container_width=True)
  #   st.line_chart(history.history['val_loss'], use_container_width=True)

with tabs[1]:
  
  canvas_result = st_canvas(stroke_width=15,
						  stroke_color='rgb(255, 255, 255)',
						  background_color='rgb(0, 0, 0)',
						  height=150,
						  width=150,
						  key="canvas")

if canvas_result.image_data is not None:
  img = canvas_result.image_data
  st.write(img.shape)
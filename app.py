import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os

st.title('Handwriting Alphabet Training')

X = []
y = []

# ds_path = 'dataset'
# folders = os.listdir(ds_path)
# for folder in folders:
#   	files = os.listdir(os.path.join(ds_path, folder))
#   	for f in files:
#     		if f.endswith('.png'):
#       			img = Image.open(os.path.join(ds_path, folder, f))
# 			img = np.array(img)
# 			X.append(img)
# 			y.append(folder)

tabs = st.tabs(["Model Training", "Drawable Canvas"])

with tabs[0]:
    	st.header("Model Training")
    	epochs = st.slider('Number of Epochs', 1, 50, 10)
    	test_size = st.slider('Test Size', 0.1, 0.5, 0.2)
    	samples_per_class = st.slider('Samples per Class', 100, 10000, 1000)
	
	if st.checkbox('View Data'):
		
	
    	if st.button('Train Model'):
        	model, history = train_model(epochs, test_size, samples_per_class)
        
        	st.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        	st.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        	st.line_chart(history.history['accuracy'], use_container_width=True)
        	st.line_chart(history.history['val_accuracy'], use_container_width=True)
        	st.line_chart(history.history['loss'], use_container_width=True)
        	st.line_chart(history.history['val_loss'], use_container_width=True)
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

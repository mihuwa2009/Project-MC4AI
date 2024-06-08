import numpy as np
from numpy import load
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os

st.title('Handwriting Alphabet Training')

X = []
y = []

# data = load('dataset.npz')
# X = data['images']
# y = data['labels']

tabs = st.tabs(["Model Training", "Drawable Canvas"])

with tabs[0]:
    	st.header("Model Training")
    	epochs = st.slider('Number of Epochs', 1, 50, 10)
    	test_size = st.slider('Test Size', 0.1, 0.5, 0.2)
    	samples_per_class = st.slider('Samples per Class', 100, 10000, 1000)
        if st.toggle('View Data'):
		fig, axs = plt.subplots(10, 10)
		fig.set_figheight(6)
		fig.set_figwidth(6)

		for i in range(26):
   			for j in range(10):
    				target = np.random.choice(np.where((y == i))[0])
    				axs[i][j].axis('off')
   				axs[i][j].imshow(X[target].reshape(28,28), cmap='gray')
	
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

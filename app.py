import streamlit as st
import numpy as np
st.title('Handwriting Alphabet Training')

epochs = st.sidebar.slider('Number of Epochs', 1, 50, 10)
test_size = st.sidebar.slider('Test Size', 0.1, 0.5, 0.2)
samples_per_class = st.sidebar.slider('Samples per Class', 100, 10000, 1000)

if st.button('Train Model'):
    model, history = train_model(epochs, test_size, samples_per_class)
    
    st.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    st.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    st.line_chart(history.history['accuracy'], use_container_width=True)
    st.line_chart(history.history['val_accuracy'], use_container_width=True)
    st.line_chart(history.history['loss'], use_container_width=True)
    st.line_chart(history.history['val_loss'], use_container_width=True)

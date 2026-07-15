import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load model
model = tf.keras.models.load_model("resnet50_age_gender_final_model.keras")

def predict_age_gender(image):
    try:
        image = image.convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)

        print("PREDICTIONS:", predictions)

        if isinstance(predictions, dict):
            age_pred = predictions["age"][0][0]
            gender_prob = predictions["gender"][0][0]
        else:
            age_pred = predictions[0][0][0]
            gender_prob = predictions[1][0][0]

        gender = "Female" if gender_prob >= 0.5 else "Male"

        return f"Predicted Age: {age_pred:.1f} years\nPredicted Gender: {gender}\nGender Probability: {gender_prob:.2f}"

    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=predict_age_gender,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Age and Gender Prediction using ResNet50",
    description="Upload a face image. The model predicts age and gender using transfer learning with ResNet50."
)

demo.launch()
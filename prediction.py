import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


MODEL_PATH = (
    "static/models/"
    "brain_tumor_mobilenetv2.keras"
)


CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# Load model once when the application starts
model = load_model(MODEL_PATH)


def predict_image(image_path):

    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    prediction = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
        * 100
    )

    probabilities = {
        CLASS_NAMES[i]:
        float(predictions[i] * 100)

        for i in range(
            len(CLASS_NAMES)
        )
    }

    return (
        prediction,
        confidence,
        probabilities
    )
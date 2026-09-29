import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


def make_gradcam(
    model,
    image_path,
    output_path,
    pred_index=None
):

    try:

        # ----------------------------------------
        # Load image
        # ----------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (224, 224)
        )

        img_array = np.array(
            image,
            dtype=np.float32
        )

        img_array = img_array / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # ----------------------------------------
        # Find MobileNetV2
        # ----------------------------------------

        base_model = None

        for layer in model.layers:

            if isinstance(
                layer,
                tf.keras.Model
            ):

                if "mobilenet" in layer.name.lower():

                    base_model = layer
                    break

        if base_model is None:

            raise ValueError(
                "MobileNetV2 base model not found."
            )

        print(
            "Base model:",
            base_model.name
        )

        # ----------------------------------------
        # Find last convolutional layer
        # ----------------------------------------

        last_conv_layer = None

        for layer in reversed(
            base_model.layers
        ):

            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            ):

                last_conv_layer = layer
                break

        if last_conv_layer is None:

            raise ValueError(
                "No Conv2D layer found."
            )

        print(
            "Grad-CAM layer:",
            last_conv_layer.name
        )

        # ----------------------------------------
        # Create Grad-CAM model
        # ----------------------------------------

        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[
                last_conv_layer.output,
                base_model.output
            ]
        )

        # ----------------------------------------
        # Forward pass
        # ----------------------------------------

        with tf.GradientTape() as tape:

            conv_outputs, _ = grad_model(
                inputs=img_array,
                training=False
            )

            # Get prediction from ORIGINAL model
            predictions = model(
                inputs=img_array,
                training=False
            )

            if pred_index is None:

                pred_index = tf.argmax(
                    predictions[0]
                )

            class_score = predictions[
                :, pred_index
            ]

        # ----------------------------------------
        # Calculate gradients
        # ----------------------------------------

        grads = tape.gradient(
            class_score,
            conv_outputs
        )

        if grads is None:

            raise ValueError(
                "Gradients could not be calculated."
            )

        # ----------------------------------------
        # Global average pooling
        # ----------------------------------------

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(1, 2)
        )

        conv_outputs = conv_outputs[0]

        pooled_grads = pooled_grads[0]

        # ----------------------------------------
        # Generate heatmap
        # ----------------------------------------

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        heatmap = heatmap / (
            tf.reduce_max(heatmap) + 1e-8
        )

        heatmap = heatmap.numpy()

        # ----------------------------------------
        # Resize heatmap
        # ----------------------------------------

        heatmap = cv2.resize(
            heatmap,
            (224, 224)
        )

        heatmap = np.uint8(
            255 * heatmap
        )

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        # ----------------------------------------
        # Original image
        # ----------------------------------------

        original = cv2.imread(
            image_path
        )

        if original is None:

            raise ValueError(
                "Could not read original image."
            )

        original = cv2.resize(
            original,
            (224, 224)
        )

        # ----------------------------------------
        # Overlay
        # ----------------------------------------

        overlay = cv2.addWeighted(
            original,
            0.60,
            heatmap_color,
            0.40,
            0
        )

        # ----------------------------------------
        # Save
        # ----------------------------------------

        folder = os.path.dirname(
            output_path
        )

        if folder:

            os.makedirs(
                folder,
                exist_ok=True
            )

        success = cv2.imwrite(
            output_path,
            overlay
        )

        if not success:

            raise ValueError(
                "Could not save Grad-CAM image."
            )

        print(
            "Grad-CAM saved:",
            output_path
        )

        return output_path

    except Exception as e:

        print(
            "Grad-CAM error:",
            e
        )

        raise
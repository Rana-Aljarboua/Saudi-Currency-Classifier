import torch
import pickle
import cv2
import numpy as np
from torchvision import transforms, models
# Path to the saved model and class names
model_save_path = "ptmodel/currency_classifier.pt"
class_names_path = "ptmodel/class_names.pkl"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved class names
with open(class_names_path, "rb") as f:
    class_names = pickle.load(f)
print("Class names loaded:", class_names)

# Define the model structure
# model = models.resnet18(pretrained=False)  # Ensure the structure matches the saved model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_save_path, map_location=device))
model = model.to(device)
model.eval()  # Set to evaluation mode
print("Model loaded successfully!")

# Transformation for input image (consistent with training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to preprocess the image using OpenCV
def preprocess_image_cv2(frame):
    # Resize the image to 224x224 as expected by the model
    resized_frame = cv2.resize(frame, (224, 224))
    # Convert the image to RGB format
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Normalize and convert to tensor
    normalized_tensor = transform(rgb_frame)
    # Add batch dimension and send to device
    return normalized_tensor.unsqueeze(0).to(device)

# Currency to value mapping (you can modify this as per your classes)
currency_values = {
    'HALALA 10': 0.10,
    'HALALA 25': 0.25,
    'HALALA 50': 0.50,
    'RIYAL 1': 1,
    'RIYAL 10 NOTE': 10,
    'RIYAL 100 NOTE': 100,
    'RIYAL 2': 2,
    'RIYAL 20 NOTE': 20,
    'RIYAL 500 NOTE': 500,
    'RIYAL 5 NOTE': 5,
    'RIYAL 50 NOTE': 50,
    'RIYAL 200 NOTE': 200
}

# Variables to store detected currencies, the running sum, and total
last_detected_currency = None
running_sum = 0.0
total = 0.0

# Function for making predictions
def predict_currency(frame, confidence_threshold=0.75):
    # Preprocess the input frame
    input_tensor = preprocess_image_cv2(frame)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence_score, predicted_class_idx = torch.max(probabilities, 1)

    # Retrieve the class name
    confidence_score = confidence_score.item()
    if confidence_score >= confidence_threshold:
        predicted_class = class_names[predicted_class_idx.item()]
        return predicted_class, confidence_score
    else:
        return "No class detected", confidence_score

# Real-time inference function
def run_inference():
    global running_sum, last_detected_currency, total
    cap = cv2.VideoCapture(1)  # Use the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform prediction on the current frame
        predicted_currency, confidence_score = predict_currency(frame)

        # If a valid, new currency is detected and is different from the last one
        if predicted_currency != "No class detected":
            if predicted_currency != last_detected_currency:  # Only proceed if it's a new classification
                if last_detected_currency is not None:
                    # Add the last detected currency and the current detected currency
                    running_sum = currency_values[last_detected_currency] + currency_values[predicted_currency]
                    total += running_sum  # Add the sum to the total

                    # Display the sum with the currency names on screen
                    print(f"Sum of {last_detected_currency} and {predicted_currency}: {running_sum}")
                    print(f"Total: {total}")

                    # Reset the sum to 0 after showing the result
                    running_sum = running_sum
                # Update the last detected currency to the current one
                last_detected_currency = predicted_currency

        # Overlay the prediction results on the frame
        if predicted_currency == "No class detected":
            overlay_text = "Not Found"
        else:
            overlay_text = f"{predicted_currency} ({confidence_score:.2f})"
        
        cv2.putText(frame, overlay_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Only display sum if a valid sum is calculated
        if running_sum >= 0:
            sum_text = f"Sum :{running_sum:.2f}"
        else:
            sum_text = f"Sum: {running_sum:.2f}"

        cv2.putText(frame, sum_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the total on the screen
        total_text = f"Total: {total:.2f}"
        cv2.putText(frame, total_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        

        # Display the video feed with predictions, sum, and total
        cv2.imshow('Currency Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the real-time inference
run_inference()

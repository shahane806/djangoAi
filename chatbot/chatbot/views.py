from django.http import JsonResponse, HttpResponse
import joblib
import pathlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import manage
from PIL import Image
import requests
import numpy as np
import tensorflow as tf

# Define the pipeline structure (not necessary for loading, but good practice)
Pipe = Pipeline([
    ('bow', CountVectorizer(analyzer=manage.cleaner)),
    ('tfidf', TfidfTransformer()),
    ('classifier', DecisionTreeClassifier())
])
filepath = pathlib.Path(__file__).resolve().parent

# Load the previously saved chatbot model
Pipe = joblib.load(filepath / 'chatbot')

def sendResponse(request):
    if request.GET.get('name') == "TSI":
        # Make a prediction
        issue_list = [
            "I have an issue",
            "Software Issues",
            "Hardware Issues",
            "Network and Connectivity",
            "Account and Access",
            "Data and Storage",
            "Security and Privacy",
            "Website and Web Application Issues",
            "System and Operating System Issues",
            "Email and Communication Issues",
            "Development and Programming Issues",
            "Installation Problems",
            "Application Crashes",
            "Performance Issues",
            "Device Connectivity",
            "Power Issues",
            "Internet Connectivity",
            "VPN Issues",
            "Login Problems",
            "Two-Factor Authentication",
            "Data Loss",
            "Storage Space",
            "Malware and Viruses",
            "Firewall Issues",
            "Page Load Errors",
            "Browser Compatibility",
            "OS Updates",
            "Driver Issues",
            "email delivery",
            "other"
        ]

        # Convert all items in the list to lowercase
        issue_list_lower = [issue.lower() for issue in issue_list]

        print(issue_list_lower)

        prediction = Pipe.predict([request.GET.get("query")])
        if request.GET.get("query").lower() not in issue_list_lower:
            prediction[0] = 'Please select a category: 1. Software Issues, 2. Hardware Issues, 3. Network and Connectivity, 4. Account and Access, 5. Data and Storage, 6. Security and Privacy, 7. Website and Web Application Issues, 8. System and Operating System Issues, 9. Email and Communication Issues, 10. Development and Programming Issues, 11. Other.'
        if request.GET.get("query").lower() == 'other':
            return HttpResponse("""
        <textarea id="elobrateInputFromDjango" type="text" placeholder="Please elaborate your issue in detail"/>
        """)
        return JsonResponse({'prediction': prediction[0]})
    elif request.GET.get("name") == "DigitRecog":
        try:
            # Get the image URL from the query parameter
            image_url = request.GET.get("query")
            # Find the starting position of 'query=' and ending position of '&'
            start = image_url.find("query=") + len("query=")
            end = image_url.find("&", start)
            if end == -1:  # Handle the case where '&' is not present
                end = len(image_url)

            # Extract and decode the URL
            encoded_url = image_url[start:end]
            from urllib.parse import unquote
            image_url = unquote(encoded_url)
            print("imageUrl : http:"+image_url)
            # image_url = "http:"+image_url
            if not image_url:
                return JsonResponse({"error": "No image URL provided. Please provide a valid image URL."}, status=400)


           
            response = requests.get("http:"+image_url, stream=True)
            print(response)
            if response.status_code != 200:
                return JsonResponse({"error": "Failed to fetch image from URL. Please check the URL."}, status=400)

            img = Image.open(response.raw)

            
            # Step 2: Resize the image to 28x28 pixels and convert to grayscale
            img = img.resize((28, 28)).convert('L')

            # Step 3: Normalize and preprocess the image
            img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            img_array = img_array.reshape(1,784)  # Reshape for the model (batch_size, height, width, channels)
            print(img_array)
            
            # Step 4: Load the pre-trained digit recognition model
            model = joblib.load(filepath / 'digit-recognization')

            # Step 5: Predict the digit
            prediction = model.predict(img_array)
            label = int(np.argmax(prediction, axis=1)[0])  # Get the digit with the highest probability

            # Step 6: Return the predicted label
            print(label)
            return HttpResponse(f"It is {label}")

        except Exception as e:
            # Handle any exceptions and provide meaningful feedback
            return HttpResponse(e)

    else:
        return HttpResponse("Something went wrong")

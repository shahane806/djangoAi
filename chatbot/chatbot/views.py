from django.http import JsonResponse,HttpResponse
import joblib
import pathlib 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import manage
# Define the pipeline structure (not necessary for loading, but good practice)
Pipe = Pipeline([
    ('bow', CountVectorizer(analyzer=manage.cleaner)),
    ('tfidf', TfidfTransformer()),
    ('classifier', DecisionTreeClassifier())
])
filepath = pathlib.Path(__file__).resolve().parent
# Load the previously saved model using the correct file name
Pipe = joblib.load(filepath/'chatbot')

def sendResponse(request):
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
    "Email Delivery",
    "Other"
]
    prediction = Pipe.predict([request.GET.get("query")])
    if request.GET.get("query") not in issue_list:
        prediction[0] = 'Please select a category: 1. Software Issues, 2. Hardware Issues, 3. Network and Connectivity, 4. Account and Access, 5. Data and Storage, 6. Security and Privacy, 7. Website and Web Application Issues, 8. System and Operating System Issues, 9. Email and Communication Issues, 10. Development and Programming Issues, 11. Other.'
    if request.GET.get("query") == 'Other':
        return HttpResponse("""
<textarea id="elobrateInputFromDjango" type="text" placeholder="Please elobrate your issue in details"/>
""")
    return JsonResponse({'prediction': prediction[0]})


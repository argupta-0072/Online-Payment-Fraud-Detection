from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model with debugging
try:
    model = pickle.load(open('payments.pkl', 'rb'))
    print("Model loaded successfully!")

    # Test the model with sample data
    test_data = np.array([[1, 4, 10000000, 1, 10000000, 0, 2, 0, 0]])
    test_prediction = model.predict(test_data)
    print(f"Test prediction: {test_prediction}")
    print(f"Model classes: {model.classes_ if hasattr(model, 'classes_') else 'No classes attribute'}")

except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/result', methods=['POST'])
def result():
    try:
        # Extract form data
        step = int(request.form['step'])
        type_val = int(request.form['type'])
        amount = float(request.form['amount'])
        nameOrig = int(request.form['nameOrig'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        nameDest = int(request.form['nameDest'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        # Prepare features
        features = np.array([[step, type_val, amount, nameOrig, oldbalanceOrg,
                              newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest]])

        print(f"Input features: {features}")  # Debug print

        # Make prediction
        if model is not None:
            prediction = model.predict(features)[0]
            print(f"Raw prediction: {prediction}")  # Debug print

            # Check what the prediction actually is
            if prediction == 1:
                result_text = "Fraud"
            elif prediction == 0:
                result_text = "Not Fraud"
            else:
                result_text = f"Unknown prediction: {prediction}"
        else:
            result_text = "Model not available"

        return render_template('result.html', prediction=result_text)

    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Debug print
        return render_template('result.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)


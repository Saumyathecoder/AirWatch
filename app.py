
# from flask import Flask, request, render_template
# import pickle
# import numpy as np

# # Load trained model
# model_path = 'pred2.pkl'
# with open(model_path, 'rb') as file:
#     model = pickle.load(file)

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Extract feature values as float
#         features = [float(x) for x in request.form.values()]
#         final_input = np.array([features])

#         # Make prediction
#         prediction = model.predict(final_input)
#         output = round(prediction[0], 4)

#         return render_template('index.html', prediction_text=f'Predicted CO(GT) value: {output}')
#     except Exception as e:
#         return render_template('index.html', prediction_text=f'Error: {str(e)}')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template, redirect, url_for, session
import pickle
import numpy as np
import os

# Load trained model
model_path = 'pred2.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')# Needed for session

@app.route('/')
def home():
    prediction_text = session.pop('prediction_text', None)  # Get and clear prediction
    return render_template('index.html', prediction_text=prediction_text)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract feature values as float
        features = [float(x) for x in request.form.values()]
        final_input = np.array([features])

        # Make prediction
        prediction = model.predict(final_input)
        output = round(prediction[0], 4)

        session['prediction_text'] = f'Predicted CO(GT) value: {output}'
        return redirect(url_for('home'))  # Redirect to clear form on refresh

    except Exception as e:
        session['prediction_text'] = f'Error: {str(e)}'
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

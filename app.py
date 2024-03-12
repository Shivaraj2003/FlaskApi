from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved Random Forest model from a .pkl file
loaded_model = joblib.load('random_forest_model.pkl')  # Replace 'your_model.pkl' with the actual filename of your model

@app.route('/')
def home():
    return "Random Forest Model Tester"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            # Handle POST request as before
            ph = float(request.form['ph'])
            temperature = int(request.form['temperature'])
            fat = float(request.form['fat'])
            colour = int(request.form['colour'])
        elif request.method == 'GET':
            # Handle GET request by extracting parameters from the URL
            ph = float(request.args.get('ph'))
            temperature = int(request.args.get('temperature'))
            fat = float(request.args.get('fat'))
            colour = int(request.args.get('colour'))
        else:
            return jsonify({'error': 'Unsupported HTTP method'}), 400

        # Create a DataFrame with user input
        new_input_data = {'pH': [ph], 'Temprature': [temperature], 'Fat ': [fat], 'Colour': [colour]}
        new_input_df = pd.DataFrame(new_input_data, columns=['pH', 'Temprature', 'Fat ', 'Colour'])

        # Ensure 'Fat' and 'Temperature' are treated as categorical variables
        try:
            new_input_df['Fat '] = new_input_df['Fat '].astype('category')
        except ValueError:
            pass  # Handle case when 'Fat' column is not present in training data

        try:
            new_input_df['Temprature'] = new_input_df['Temprature'].astype('category')
        except ValueError:
            pass  # Handle case when 'Temperature' column is not present in training data

        # Ensure the DataFrame has the same columns as the training data
        expected_columns = ['pH', 'Temprature', 'Fat ', 'Colour']
        missing_columns = set(expected_columns) - set(new_input_df.columns)

        if missing_columns:
            return jsonify({'error': f"Missing columns in input data: {missing_columns}"}), 400

        # Make predictions using the loaded Random Forest model
        y_pred_loaded_model = loaded_model.predict(new_input_df)

        # Display the predicted grade
        predicted_grade = int(y_pred_loaded_model[0])  # Convert to int

        return jsonify({'predicted_grade': predicted_grade})

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400  # Bad Request
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500  # Internal Server Error

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

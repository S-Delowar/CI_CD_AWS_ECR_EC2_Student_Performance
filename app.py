from flask import Flask, request, render_template
import pickle, os
import pandas as pd

from src.logger import logging
from src.utils import load_object


application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_math_score():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # If the method is POST
        # Steps:
                # 1. Get the input values
                # 2. Convert to DataFrame with column name. Column names & serial must be same to the main dataset.
                        # two way to convert into DataFrame:
                        #     1. Convert the input data to list and then to DataFrame with setting column names
                        #     2. Creating Dictionary. key=> column name, value=> input data point as list
                        #         like- {"gender": [request.form.get("gender")], "lunch": [request.form.get("lunch")]}
                        #     3. Create CustomData class in the utils.py and create a method to return dataFrame. 
                        #         In the    method - create a dict like {"gender":[self.gender], ......}, then to DataFrame 
                         
                    # I prefer the 2nd way 
                # 3. load the model and preprocessor as readline => with open(file_path, 'rb') as ........
                # 4. transform the input dataFrame by the preprocessor
                # 5. predict the result
                # 6. result type arr[] => predicted_value = result[0]
        
        
        # Get the value from the form and store in a dictionary to convert into dataFrame
        # request.form.get(value here as same as the "name" attribute in input in the form)
        input_dict = {
            'gender' : [request.form.get('gender')],
            'race_ethnicity' : [request.form.get('race_ethnicity')],
            'parental_level_of_education' : [request.form.get("parental_level_of_education")],
            'lunch' : [request.form.get("lunch")],
            'test_preparation_course': [request.form.get("test_preparation_course")],
            'reading_score' : [float(request.form.get("reading_score"))],
            'writing_score' : [float(request.form.get("writing_score"))]
        }
        
        # Convert the dictionary into pandas DataFrame
        input_df = pd.DataFrame(input_dict)
        
        # load the model and preprocessor        
        model_path = os.path.join('artifacts', 'model.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        
        # process the input data
        processed_input_df = preprocessor.transform(input_df)
        
        prediction = model.predict(processed_input_df)
        math_score = round(prediction[0])
    
        return render_template('home.html', math_score = math_score)
            
        
    
if __name__ == '__main__':
    app.debug=True
    app.run()
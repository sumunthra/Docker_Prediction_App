import re
import joblib
import gradio
import gradio as gr
import pandas as pd


# Load model
trained_model = joblib.load(filename = "titanic-survival-pred-model.pkl")


# Extracts the title (Mr, Ms, etc) from the name variable
def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'


# UI - Input components
in_Pid = gradio.Textbox(lines=1, placeholder=None, value="79", label='Passenger Id')
in_Pclass = gradio.Radio([1, 2, 3], type="value", label='Passenger class')
in_Pname = gradio.Textbox(lines=1, placeholder=None, value="Caldwell, Master. Alden Gates", label='Passenger Name')
in_sex = gradio.Radio(["Male", "Female"], type="value", label='Gender')
in_age = gradio.Textbox(lines=1, placeholder=None, value="14", label='Age of the passenger in yrs')
in_sibsp = gradio.Textbox(lines=1, placeholder=None, value="0", label='No. of siblings/spouse of the passenger aboard')
in_parch = gradio.Textbox(lines=1, placeholder=None, value="2", label='No. of parents/children of the passenger aboard')
in_ticket = gradio.Textbox(lines=1, placeholder=None, value="248738", label='Ticket number')
in_cabin = gradio.Radio(["Yes", "No"], type="value", label='Has Cabin')
# in_cabin = gradio.Textbox(lines=1, placeholder=None, value="A5", label='Cabin number')
in_embarked = gradio.Radio(["Southampton", "Cherbourg", "Queenstown"], type="value", label='Port of Embarkation')
in_fare = gradio.Textbox(lines=1, placeholder=None, value="29", label='Passenger fare')

# UI - Output component
out_label = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox")


# Mappings for categorical features
embarked_mapping = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}
gender_mapping = {'Female': 0, 'Male': 1}
title_mapping = {'Mrs': 4, 'Master':3 ,'Miss': 2, 'Mr': 1,'Other':0}
cabin_mapping = {"Yes": 1, "No": 0}


# Label prediction function
def get_output_label(in_Pid, in_Pclass, in_Pname, in_sex, in_age, in_sibsp, in_parch, in_ticket, in_cabin, in_embarked, in_fare):
    
    input_df = pd.DataFrame({'Pclass': [in_Pclass], 
                             'Sex': [gender_mapping[in_sex]], 
                             'Age': [in_age],
                             'Fare': [in_fare],
                             'Embarked': [embarked_mapping[in_embarked]],
                             'FamilySize': [int(in_sibsp) + int(in_parch) + 1], 
                             'Has_cabin': [cabin_mapping[in_cabin]], 
                             'Title': [title_mapping[get_title(in_Pname)]]})
    
    prediction = trained_model.predict(input_df)     # Make a prediction using the saved model
    if prediction[0] == 1:
        label = "Likely to Survive"
    else:
        label = "Less likely to Survive"

    return label


# Create Gradio interface object
iface = gradio.Interface(fn = get_output_label,
                         inputs = [in_Pid, in_Pclass, in_Pname, in_sex, in_age, in_sibsp, in_parch, in_ticket, in_cabin, in_embarked, in_fare],
                         outputs = [out_label],
                         title="Titanic Survival Prediction API ⛴",
                         description="Predictive model that answers the question: “What sort of people were more likely to survive?”",
                         flagging_mode='never'
                         )

# Launch gradio interface
iface.launch(server_name = "0.0.0.0", server_port = 7860)
                         # set server_name = "0.0.0.0" and server_port = 7860 while launching it inside container.
                         # default server_name = "127.0.0.1", and server_port = 7860

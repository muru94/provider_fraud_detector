from flask import Flask, jsonify, request
import pandas as pd
import re
from icd9cms.icd9 import search
import scipy
from scipy.sparse import hstack
import joblib
import django_heroku


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

bow_vect = joblib.load('Bow_vectorizer.pkl')
Inscclmreimburse_scaler = joblib.load('Inscclmreimburse_scaler.pkl')
clmadmitdiag_scaler = joblib.load('clmadmitdiag_scaler.pkl')
DeductibleAmtPaid_scaler = joblib.load('DeductibleAmtPaid_scaler.pkl')
DiagnosisGroupCode_scaler = joblib.load('DiagnosisGroupCode_scaler.pkl')
ClmDiagnosisCode_scaler = joblib.load('ClmDiagnosisCode_scaler.pkl')
ClmProcCode_scaler = joblib.load('ClmProcCode_scaler.pkl')
race_scaler = joblib.load('race_scaler.pkl')
state_scaler = joblib.load('state_scaler.pkl')
county_scaler = joblib.load('county_scaler.pkl')
Annualreimburse_scaler = joblib.load('Annualreimburse_scaler.pkl')
Annualdeduct_scaler = joblib.load('Annualdeduct_scaler.pkl')
patientage_scaler = joblib.load('patientage_scaler.pkl')
treatmentdays_scaler = joblib.load('treatmentdays_scaler.pkl')
treatmentmonth_scaler = joblib.load('treatmentmonth_scaler.pkl')
treatmentdate_scaler = joblib.load('treatmentdate_scaler.pkl')
noofdiagcode_scaler = joblib.load('noofdiagcode_scaler.pkl')
noofproccode_scaler = joblib.load('noofproccode_scaler.pkl')
sumchronic_scaler = joblib.load('sumchronic_scaler.pkl')
noofphysiciansconsult_scaler = joblib.load('noofphysiciansconsult_scaler.pkl')
model = joblib.load('model.pkl')

def final_predict(test_dict):
    
    # International Statistical Classification of Diseases and Related Health Problems (ICD), a medical classification list by the World Health Organization
    # There is an unique ICD codes for each disease. 
    # As we given ClmDiagnosisCode_1 to 10 - 10 diagnosis code for each claims. We are extracting the description of the diagnosis code.
    # Extraction of the diagnosis desciption can be done using icd9.cms pre-defined library
    # Example : search('Diagnosis code#') -> returns the desc - search('4019') -> 4019:Hypertension NOS:Unspecified essential hypertension

    def fetch_diagnosis_des(test_dic, diagnosis):
        tempstr = ''
        for i in diagnosis:
            try:
                code = search(test_dic.get(i))
                code = str(code)
                tempstr = tempstr + " " + code.split(":")[2]
            except:
                continue
        return tempstr
    
    diagnosisfields = ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7','ClmDiagnosisCode_8','ClmDiagnosisCode_9','ClmDiagnosisCode_10']
    DiagnosisDesc = fetch_diagnosis_des(test_dict, diagnosisfields)
        
    def decontracted(phrase):
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        
        return phrase

    # https://gist.github.com/sebleier/554280
    # we are removing the words from the stop words list: 'no', 'nor', 'not'

    stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
    
    # Combining all the above steps 
    def preprocessing(textdata):
        codedescrib = list()
        for sentance in textdata:
            sentance = decontracted(sentance)
            #remove words with numbers python: https://stackoverflow.com/a/18082370/4084039
            sentance = re.sub("\S*\d\S*", "", sentance).strip()
            #remove spacial character: https://stackoverflow.com/a/5843547/4084039
            sentance = re.sub('[^A-Za-z]+', ' ', sentance)
            # https://gist.github.com/sebleier/554280
            sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
            codedescrib.append(sentance.strip())
        return codedescrib
    
    preprocessed_dia = preprocessing([DiagnosisDesc])
    
    
    test_RenalDiseaseIndicator = 1 if test_dict.get('RenalDiseaseIndicator') == 'Y' else 0
    
    test_dodind = 0 if test_dict.get('DOD') == '0' else 1
    
    # Calculating the patient age
    test_dob = pd.to_datetime(test_dict.get('DOB'))
    patientage = 2009 - int(str(test_dob).split("-")[0])
    
    startdate = pd.to_datetime(test_dict.get('ClaimStartDt'))
    enddate = pd.to_datetime(test_dict.get('ClaimEndDt'))
    treatmentdays = enddate - startdate
    treatmentdays = int(treatmentdays.days)
    
    # Finding the treatment month & last treatment date
    treatmentmonth = int(str(enddate).split("-")[1])
    treatmentdate = int((str(enddate).split("-")[2]).split(" ")[0])
   
    # to calculate the no of diagnosis code
    def count(test_d, codes):
        count = 0
        for i in codes:
            if test_d.get(i) == '0':
                continue
            else:
                count += 1
        return count
    
    NoofDigcode = count(test_dict, diagnosisfields)
    
    # to calculate no of proc codes used for the claim
    proccodes = ['ClmProcedureCode_1', 'ClmProcedureCode_2','ClmProcedureCode_3']
    NoofProccode = count(test_dict, proccodes)
    
    # to calculate the summation of chronic codition of diseases
    
    def sum_chronic(test_d, codes):
        count = 0
        for i in codes:
            if test_d.get(i) == '0':
                continue
            else:
                count = count + int(test_d.get(i))
        return count
    
    chronicdisease = ['ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
                 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
                 'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke']
    
    sumchronic = sum_chronic(test_dict, chronicdisease)
    
    # Calculating no of physicians consulted
    physicians = ['AttendingPhysician','OperatingPhysician', 'OtherPhysician']
    noofphysiciansconsulted = count(test_dict, physicians)
    
    # Set 1 if there is a physician
    AttendingPhysician = int(test_dict.get('AttendingPhysician'))
    OperatingPhysician = int(test_dict.get('OperatingPhysician'))
    OtherPhysician = int(test_dict.get('OtherPhysician'))
    
    def convert(test_d, code):
        try:
            return int(''.join(e for e in test_d.get(code) if e.isdigit()))
        except:
            return 0
    
    # Remove the characters from the Diagnosis codes
    ClmAdmitDiagnosisCode = convert(test_dict, 'ClmAdmitDiagnosisCode')
    
    # Remove the characters from the Diagnosis Group codes
    DiagnosisGroupCode = convert(test_dict, 'DiagnosisGroupCode')
    
    # Remove the characters from the claim diagnosis codes
    ClmDiagnosisCode_1 = convert(test_dict, 'ClmDiagnosisCode_1')
    
    ClmDiagnosisCode_2 = convert(test_dict, 'ClmDiagnosisCode_2')
    
    ClmDiagnosisCode_3 = convert(test_dict, 'ClmDiagnosisCode_3')
    
    ClmDiagnosisCode_4 = convert(test_dict, 'ClmDiagnosisCode_4')
    
    ClmDiagnosisCode_5 = convert(test_dict, 'ClmDiagnosisCode_5')
    
    ClmDiagnosisCode_6 = convert(test_dict, 'ClmDiagnosisCode_6')
    
    ClmDiagnosisCode_7 = convert(test_dict, 'ClmDiagnosisCode_7')
    
    ClmDiagnosisCode_8 = convert(test_dict, 'ClmDiagnosisCode_8')
    
    ClmDiagnosisCode_9 = convert(test_dict, 'ClmDiagnosisCode_9')
    
    ClmDiagnosisCode_10 = convert(test_dict, 'ClmDiagnosisCode_10')
    

    # mapping Chronic condition 1 to 0 and 2 to 1
    ChronicCond_Alzheimer = 0 if test_dict.get('ChronicCond_Alzheimer') == '1' else 1
    ChronicCond_Heartfailure = 0 if test_dict.get('ChronicCond_Heartfailure') == '1' else 1 
    ChronicCond_KidneyDisease = 0 if test_dict.get('ChronicCond_KidneyDisease') == '1' else 1 
    ChronicCond_Cancer = 0 if test_dict.get('ChronicCond_Cancer') == '1' else 1  
    ChronicCond_ObstrPulmonary = 0 if test_dict.get('ChronicCond_ObstrPulmonary') == '1' else 1 
    ChronicCond_Depression = 0 if test_dict.get('ChronicCond_Depression') == '1' else 1 
    ChronicCond_Diabetes = 0 if test_dict.get('ChronicCond_Diabetes') == '1' else 1 
    ChronicCond_IschemicHeart = 0 if test_dict.get('ChronicCond_IschemicHeart') == '1' else 1 
    ChronicCond_Osteoporasis = 0 if test_dict.get('ChronicCond_Osteoporasis') == '1' else 1 
    ChronicCond_rheumatoidarthritis = 0 if test_dict.get('ChronicCond_rheumatoidarthritis') == '1' else 1
    ChronicCond_stroke = 0 if test_dict.get('ChronicCond_stroke') == '1' else 1
    
    Gender = 0 if test_dict.get('Gender') == '2' else 1 # mapping gender code 2->0 and 1->1
    
    # loading the pre-trained Countvectorizer and transform on test data
    test_diagnosisdesc = bow_vect.transform(preprocessed_dia)
    
    test_Insreimburse = Inscclmreimburse_scaler.transform([[int(test_dict.get('InscClaimAmtReimbursed'))]])
    
    test_admitdiagnosis = clmadmitdiag_scaler.transform([[ClmAdmitDiagnosisCode]])
    
    test_deductamt = DeductibleAmtPaid_scaler.transform([[int(test_dict.get('DeductibleAmtPaid'))]])
    
    test_diagnosisgroup = DiagnosisGroupCode_scaler.transform([[DiagnosisGroupCode]])
    
    test_diagcode1 = ClmDiagnosisCode_scaler.transform([[ClmDiagnosisCode_1]])
    
    test_diagcode2 = ClmDiagnosisCode_scaler.transform([[ClmDiagnosisCode_2]])
    
    test_diagcode3 = ClmDiagnosisCode_scaler.transform([[ClmDiagnosisCode_3]])
    
    test_diagcode4 = ClmDiagnosisCode_scaler.transform([[ClmDiagnosisCode_4]])
    
    test_diagcode5 = ClmDiagnosisCode_scaler.transform([[ClmDiagnosisCode_5]])
    
    test_diagcode6 = ClmDiagnosisCode_scaler.transform([[ClmDiagnosisCode_6]])
    
    test_diagcode7 = ClmDiagnosisCode_scaler.transform([[ClmDiagnosisCode_7]])
    
    test_diagcode8 = ClmDiagnosisCode_scaler.transform([[ClmDiagnosisCode_8]])
    
    test_diagcode9 = ClmDiagnosisCode_scaler.transform([[ClmDiagnosisCode_9]])
    
    test_diagcode10 = ClmDiagnosisCode_scaler.transform([[ClmDiagnosisCode_10]])
    
    test_proccode1 = ClmProcCode_scaler.transform([[int(test_dict.get('ClmProcedureCode_1'))]])
    
    test_proccode2 = ClmProcCode_scaler.transform([[int(test_dict.get('ClmProcedureCode_2'))]])
    
    test_proccode3 = ClmProcCode_scaler.transform([[int(test_dict.get('ClmProcedureCode_3'))]])
    
    test_race = race_scaler.transform([[int(test_dict.get('Race'))]])
    
    test_state = state_scaler.transform([[int(test_dict.get('State'))]])
    
    test_county = county_scaler.transform([[int(test_dict.get('County'))]])
    
    test_annualreimburse = Annualreimburse_scaler.transform([[int(test_dict.get('AnnualReimbursementAmt'))]])
    
    test_annualdeduct = Annualdeduct_scaler.transform([[int(test_dict.get('AnnualDeductibleAmt'))]])
    
    test_patientage = patientage_scaler.transform([[patientage]])
    
    test_treatmentdays = treatmentdays_scaler.transform([[treatmentdays]])
    
    test_treatmentmonth = treatmentmonth_scaler.transform([[treatmentmonth]])
    
    test_treatmentdate = treatmentdate_scaler.transform([[treatmentdate]])
    
    test_noofdigcode = noofdiagcode_scaler.transform([[NoofDigcode]])
    
    test_noofproccode = noofproccode_scaler.transform([[NoofProccode]])
    
    test_sumchronic = sumchronic_scaler.transform([[sumchronic]])
    
    test_noofpysicians = noofphysiciansconsult_scaler.transform([[noofphysiciansconsulted]])
    
    test_gender = scipy.sparse.csr_matrix(Gender)
    test_RenalDiseaseIndicator = scipy.sparse.csr_matrix(test_RenalDiseaseIndicator)
    test_dodind = scipy.sparse.csr_matrix(test_dodind)
    test_chronic1 = scipy.sparse.csr_matrix(ChronicCond_Alzheimer)
    test_chronic2 = scipy.sparse.csr_matrix(ChronicCond_Heartfailure)
    test_chronic3 = scipy.sparse.csr_matrix(ChronicCond_KidneyDisease)
    test_chronic4 = scipy.sparse.csr_matrix(ChronicCond_Cancer)
    test_chronic5 = scipy.sparse.csr_matrix(ChronicCond_ObstrPulmonary)
    test_chronic6 = scipy.sparse.csr_matrix(ChronicCond_Depression)
    test_chronic7 = scipy.sparse.csr_matrix(ChronicCond_Diabetes)
    test_chronic8 = scipy.sparse.csr_matrix(ChronicCond_IschemicHeart)
    test_chronic9 = scipy.sparse.csr_matrix(ChronicCond_Osteoporasis)
    test_chronic10 = scipy.sparse.csr_matrix(ChronicCond_rheumatoidarthritis)
    test_chronic11 = scipy.sparse.csr_matrix(ChronicCond_stroke)
    
    # Mergini all text vectorizer & numerical values using hstack
    test_comb = hstack((test_diagnosisdesc, test_Insreimburse, test_admitdiagnosis, test_deductamt, test_diagnosisgroup, test_diagcode1, test_diagcode2, test_diagcode3,
                       test_diagcode4, test_diagcode5, test_diagcode6, test_diagcode7, test_diagcode8, test_diagcode9, test_diagcode10, test_proccode1, test_proccode2,
                        test_proccode3, test_race, test_state, test_county, test_annualreimburse, test_annualdeduct, test_patientage, test_treatmentdays, test_treatmentmonth, 
                        test_treatmentdate, test_noofdigcode, test_noofproccode, test_sumchronic, test_noofpysicians, test_gender, test_RenalDiseaseIndicator,
                        test_dodind, test_chronic1, test_chronic2, test_chronic3, test_chronic4, test_chronic5, test_chronic6, test_chronic7,
                        test_chronic8, test_chronic9, test_chronic10, test_chronic11)).tocsr()
    
    y_pred = model.predict(test_comb) # Predicting provider fraudulent status for test data
    
    return y_pred



@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/provider')
def index():
    return flask.render_template('provider.html')


@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    ProviderID  = str(to_predict_list['Provider'])
    dia1  = str(to_predict_list['ClmDiagnosisCode_1'])
    dia2  = str(to_predict_list['ClmDiagnosisCode_2'])
    dia3  = str(to_predict_list['ClmDiagnosisCode_3'])
    dia4  = str(to_predict_list['ClmDiagnosisCode_4'])
    dia5  = str(to_predict_list['ClmDiagnosisCode_5'])
    dia6  = str(to_predict_list['ClmDiagnosisCode_6'])
    dia7  = str(to_predict_list['ClmDiagnosisCode_7'])
    dia8  = str(to_predict_list['ClmDiagnosisCode_8'])
    dia9  = str(to_predict_list['ClmDiagnosisCode_9'])
    dia10  = str(to_predict_list['ClmDiagnosisCode_10'])
    renal  = str(to_predict_list['RenalDiseaseIndicator'])
    dod  = str(to_predict_list['DOD'])
    dob  = str(to_predict_list['DOB'])
    sex  = str(to_predict_list['Gender'])
    racecd  = str(to_predict_list['Race'])
    statecd  = str(to_predict_list['State'])
    countycd  = str(to_predict_list['County'])
    clmstart  = str(to_predict_list['ClaimStartDt'])
    clmend  = str(to_predict_list['ClaimEndDt'])
    proc1  = str(to_predict_list['ClmProcedureCode_1'])
    proc2  = str(to_predict_list['ClmProcedureCode_2'])
    proc3  = str(to_predict_list['ClmProcedureCode_3'])
    chralz  = str(to_predict_list['ChronicCond_Alzheimer'])
    chrhea  = str(to_predict_list['ChronicCond_Heartfailure'])
    chrkid  = str(to_predict_list['ChronicCond_KidneyDisease'])
    chrcan  = str(to_predict_list['ChronicCond_Cancer'])
    chrobs  = str(to_predict_list['ChronicCond_ObstrPulmonary'])
    chrdep  = str(to_predict_list['ChronicCond_Depression'])
    chrdia  = str(to_predict_list['ChronicCond_Diabetes'])
    chrisc  = str(to_predict_list['ChronicCond_IschemicHeart'])
    chrost  = str(to_predict_list['ChronicCond_Osteoporasis'])
    chrrhe  = str(to_predict_list['ChronicCond_rheumatoidarthritis'])
    chrstr  = str(to_predict_list['ChronicCond_stroke'])
    atphy  = str(to_predict_list['AttendingPhysician'])
    opphysi  = str(to_predict_list['OperatingPhysician'])
    othphysi  = str(to_predict_list['OtherPhysician'])
    admitdiag  = str(to_predict_list['ClmAdmitDiagnosisCode'])
    diaggrp  = str(to_predict_list['DiagnosisGroupCode'])
    insreimburse  = str(to_predict_list['InscClaimAmtReimbursed'])
    dedamt  = str(to_predict_list['DeductibleAmtPaid'])
    annualreimb  = str(to_predict_list['AnnualReimbursementAmt'])
    annualdeduct  = str(to_predict_list['AnnualDeductibleAmt'])

    test_dict_1 = {'Provider': ProviderID, 'ClmDiagnosisCode_1' : dia1, 'ClmDiagnosisCode_2' : dia2, 'ClmDiagnosisCode_3' : dia3, 
             'ClmDiagnosisCode_4' : dia4, 'ClmDiagnosisCode_5' : dia5, 'ClmDiagnosisCode_6' : dia6,
             'ClmDiagnosisCode_7' : dia7, 'ClmDiagnosisCode_8' : dia8, 'ClmDiagnosisCode_9' : dia9,
             'ClmDiagnosisCode_10' : dia10,'RenalDiseaseIndicator' : renal, 'DOD' : dod, 'DOB' : dob,
             'Gender' : sex, 'Race' : racecd, 'State' : statecd, 'County' : countycd, 'ClaimStartDt' : clmstart,
             'ClaimEndDt' : clmend, 'ClmProcedureCode_1' : proc1, 'ClmProcedureCode_2' : proc2,
             'ClmProcedureCode_3' : proc3,'ChronicCond_Alzheimer' : chralz, 'ChronicCond_Heartfailure' : chrhea, 
             'ChronicCond_KidneyDisease' : chrkid, 'ChronicCond_Cancer' : chrcan, 'ChronicCond_ObstrPulmonary' : chrobs,
             'ChronicCond_Depression' : chrdep, 'ChronicCond_Diabetes' : chrdia, 'ChronicCond_IschemicHeart' : chrisc,
             'ChronicCond_Osteoporasis' : chrost, 'ChronicCond_rheumatoidarthritis' : chrrhe, 'ChronicCond_stroke' : chrstr,
             'AttendingPhysician' : atphy, 'OperatingPhysician' : opphysi, 'OtherPhysician' : othphysi, 
             'ClmAdmitDiagnosisCode' : admitdiag, 'DiagnosisGroupCode' : diaggrp, 'InscClaimAmtReimbursed' : insreimburse, 
             'DeductibleAmtPaid' : dedamt,'AnnualReimbursementAmt' : annualreimb, 'AnnualDeductibleAmt' : annualdeduct}
    
    pred = final_predict(test_dict_1)
    if pred == 1:
        prediction = "Potential fraudulent status of the provider# " + ProviderID + " is Yes"
    else:
        prediction = "Potential fraudulent status of the provider# " + ProviderID + " is No"

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run()
django_heroku.settings(locals())

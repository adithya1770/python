from flask import Flask, render_template, request
import pandas as pd
import spacy
import json, random

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def homepage():
    if request.method=='POST':
        promt_msg = request.form['prompt']
        nlp = spacy.load("en_core_web_sm")
        dataset = pd.read_csv('./recipes_data/recipes_data.csv')
        inputPrompt = promt_msg
        tokenizedData = nlp(inputPrompt)
        ingredientToken = list()
        matched = list()
        for i in tokenizedData:
            if i.pos_ == "NOUN":
                ingredientToken.append(i.text)
            else:
                continue
        count = 0
        for index, rows in dataset.iterrows():
            dataset_json = json.loads(rows['ingredients'])
            for i in ingredientToken:
                if i in dataset_json:
                    matched.append(index)
                    count+=1
            if count > 10:
                break
            print(matched)
        matched_rand = random.choice(matched)
        req_op = dataset.iloc[matched_rand]
    return render_template("home.html", prompt_resp=req_op)
    
if __name__ == '__main__':
    app.run(debug=True)
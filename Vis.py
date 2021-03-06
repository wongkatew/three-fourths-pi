import argparse
import io
from autocorrect import spell
import sys
import json
from google.cloud import vision
import base64

from flask import Flask, jsonify, render_template, request
app = Flask(__name__)

@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a + b)

@app.route('/')
def index():
    return render_template('index.html')


img_data = sys.argv[1]
with open("imageToSave.png", "wb") as fh:
    fh.write(base64.decodebytes(img_data))

def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    #print('Texts:')
   # print('\n"{}"'.format(texts[0].description))
    string = texts[0].description
    #print(string)
  #  print()
    #print("String: " + string)

   # for i in range(len(texts)):
 #       print(texts[i].description)
    return string
        

#print("Test Notebook stuff:")

string = detect_text("imageToSave.png")

'''
print("--------")
detect_text("image2.jpeg")
print("--------")
detect_text("image3.jpeg")'''

string = string.split('\n')
string = ' '.join(string)
string = string.split(' ')
string = [value for value in string if value != '|']
print(string)


class_type = ["LE", "DI", "FI", "ST", "SE"]
days = ["MWF", "TuTh", "M", "Tu", "W", "Th", "F"]

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


if "LE" in string:
    object_list = []
    data = {}
    counter = 0
    i = 0
    class_label = ""
    while i < len(string):
        if string[i] == 'Exam':
            object_list.append(data)
            data = {}
            data['class'] = class_label
            data['type'] = "Fl"

        if string[i] == "Enrolled":
            object_list.append(data)
            data = {}
            data['class'] = class_label

        if counter == 0:
            class_label = string[i] + string[i+1]
            data['class'] = class_label
            counter += 1
            i+=1
        if counter == 1:
            if string[i] in class_type:
                data['type'] = string[i]
                counter += 1
        if string[i] in days:
            data['day'] = string[i]
            counter += 1

        if "/" in string[i] and len(string[i]) < 15:
            data["date"] = string[i]
            i+=1

        if "p-" in string[i] or "a-" in string[i]:
            temp = string[i]
            if (len(temp) >= 15):
                data['date'] = temp[:10]
                temp = temp[10:]
            data['time'] = temp
            i+=1
            if hasNumbers(string[i]):
                head = string[i].rstrip('0123456789')
                tail = string[i][len(head):]
                data["room"] = head
                data["number"] = tail

            else:
                data["room"] = string[i]
                i+=1
                data["number"] = string[i]

        if string[i] == "TBA":
            object_list.append(data)
            data = {}
            counter = 0;

        i+=1
            
    with open('data.txt', 'w') as outfile:
        outfile.write("data = '")
        json.dump(object_list, outfile)
        outfile.write("';")
else:
    string = ' '.join(string)
    with open('data.txt', 'w') as outfile:
        outfile.write(string)



import argparse
import io
from autocorrect import spell
import sys

from google.cloud import vision

import nltk
nltk.download('words')
from nltk.corpus import words
word_list = words.words()
word_set = set(word_list)


def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    print('Texts:')
    print('\n"{}"'.format(texts[0].description))
    string = texts[0].description
    print()
    print("String: " + string)

    first = True
    for i in range(len(texts)):
        #print(texts[i])
        if first:
            first = False
            continue

        print(spell(texts[i].description))
        '''if spell(texts[i].description) in word_set:
            print(spell(texts[i].description))
        else:
            new_word = spell(texts[i].description+texts[i+1].description)
            if new_word in word_set:
                print(new_word)'''

    #print(spell(texts[0].description))
    #for text in texts:
        #print('\n"{}"'.format(text.description))

        #vertices = (['({},{})'.format(vertex.x, vertex.y)
        #            for vertex in text.bounding_poly.vertices])

        #print('bounds: {}'.format(','.join(vertices)))
        
    #for text in texts:
     #   print(text.description)

#detect_text("test2.jpg")

print("Test Notebook stuff:")

detect_text(sys.argv[1])
'''
print("--------")
detect_text("image2.jpeg")
print("--------")
detect_text("image3.jpeg")'''
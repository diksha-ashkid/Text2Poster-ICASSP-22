# -*- encoding: utf-8 -*-
'''
@File    :   quick_start.py
@Time    :   2023/01/16 23:15:11
@Author  :   Chuhao Jin
@Email   :   jinchuhao@ruc.edu.cn
'''

# here put the import lib
import os, time, json, requests
timestamp = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())) 

input_text_elements = {
    "sentences": [
        ["CHILDREN'S DAY", 90], # [text, font_fize]
        ["Children are The Future of Nation", 50] # [text, font_fize]
    ],
    "background_query": "Children's Day!" # sentence used to retrieve background images.
}

input_text_elements = json.dumps(input_text_elements)
api_url = "http://1.13.255.9:8889/text2poster"
response = requests.get(api_url, params = {"input_text_elements": input_text_elements})
f = open("poster-{}.jpg".format(timestamp), "wb")
f.write(response.content)
f.close()
print("Save poster to:", "poster-{}.jpg".format(timestamp))
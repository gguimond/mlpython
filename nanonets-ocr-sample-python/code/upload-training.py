import os, requests, json
from tqdm import tqdm

pathToAnnotations = './annotations/json'
pathToImages = './images'
os.environ['NANONETS_MODEL_ID'] = 'd8b67eee-3fbb-4064-9adc-93c242d4c3d1'
os.environ['NANONETS_API_KEY'] = 'yzwmDDQAx3z6kDNKmk6y1LmR2-Ae-UJ4'
model_id = os.environ.get('NANONETS_MODEL_ID')
api_key = os.environ.get('NANONETS_API_KEY')

for root, dirs, files in os.walk(pathToAnnotations, topdown=False):
    for name in tqdm(files):
        annotation = open(os.path.join(root, name), "r")
        filePath = os.path.join(root, name)
        imageName, ext = name.split(".")
        if imageName == "":
            continue
        imagePath = os.path.join(pathToImages, imageName + '.jpg')
        jsonData = annotation.read()
        url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + model_id + '/UploadFile/'
        data = {'file' :open(imagePath, 'rb'),  'data' :('', '[{"filename":"' + imageName+".jpg" + '", "object": '+ jsonData+'}]'),   'modelId' :('', model_id)}       
        response = requests.post(url, auth=requests.auth.HTTPBasicAuth(api_key, ''), files=data)
        if response.status_code > 250 or response.status_code<200:
            print(response.text), response.status_code

print("\n\n\nNEXT RUN: python ./code/train-model.py")

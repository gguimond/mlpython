import requests, os

os.environ['NANONETS_MODEL_ID'] = '14da5702-cab8-40a8-be57-42147da82580'
os.environ['NANONETS_API_KEY'] = 'yzwmDDQAx3z6kDNKmk6y1LmR2-Ae-UJ4'

model_id = os.environ.get('NANONETS_MODEL_ID')
api_key = os.environ.get('NANONETS_API_KEY')

url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + model_id + '/Train/'

querystring = {'modelId': model_id}

response = requests.request('POST', url, auth=requests.auth.HTTPBasicAuth(api_key, ''), params=querystring)

print(response.text)

print("\n\nNEXT RUN: python ./code/model-state.py")
import json

with open("SimController.json", "rb") as f:
    json_data = json.load(f)



print(json_data['Got'])

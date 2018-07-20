
# import the necessary packages
import requests
import sys


if __name__ == '__main__':
	# initialize the Keras REST API endpoint URL along with the input image path
	url = "http://localhost:5000/predict"
	csv_path = sys.argv[1]

	# load the input image and construct the payload for the request
	csv_file = open(csv_path, "rb").read()
	payload = {"csv": csv_file}

	# submit the request
	res = requests.post(url, files=payload).json()

	# ensure the request was sucessful
	if res["success"]:
		print(res["predictions"])
	else:
		print("Request failed")

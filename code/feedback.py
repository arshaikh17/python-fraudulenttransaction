
# import the necessary packages
import requests
import sys


if __name__ == '__main__':
	# initialize the Keras REST API endpoint URL along with the input
	# result {"true", "false"}
	url = "http://localhost:5000/feedback"
	result = sys.argv[1]

	# set the feedback of prediction and construct the data for the request
	data = {"feedback": result}
	res = requests.post(url, data=data).json()

	# ensure the request was sucessful
	if res["success"]:
		# loop over the predictions and display them
		print(res["detail"])
	else:
		print("Request failed")

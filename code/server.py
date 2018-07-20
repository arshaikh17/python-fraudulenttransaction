
# import the necessary packages
from keras.models import load_model
import flask
import csv


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

# load detect model
model_file = 'fraud_detect_model.h5'
model = load_model(model_file)


# fraud detection routine
@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the view
	data = {}

	# ensure an .CSV file was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get('verify'):
			reader = csv.DictReader(flask.request.files['verify'])
			data["predictions"] = []

			# check if the model is on the server side
			if model == None:
				return flask.jsonify({
					"error": "model doesn't exist", 
					"success": False
				})

			# check all records to check if it is fraudulent or not
			idx = 0
			for row in reader:
				# classify the input image and then initialize the list
				# of predictions to return to the client
				label = model.predict_classes(row)
				proba = model.predict_proba(row)
				class_id = label[0]
				idx = idx + 1

				# loop over the results and add them to the list of returned predictions
				record = {
					"target": idx,
					"result": class_id
					"probability": float(proba[0][class_id]),
					"model_file": model_file,
				}
				data["predictions"].append(record)

			# indicate that the request was a success
			data["success"] = True
		else:
			return flask.jsonify({
				"error": "csv file is not uploaded", 
				"success": False
			})

	# return the data dictionary as a JSON response
	return flask.jsonify(data)


# feedback option
@app.route("/feedback", methods=["POST"])
def feedback():
	# initialize the data dictionary that will be returned from the view

	if flask.request.method == "POST":
		data = {"result": False}
		data["detail"] = "Thanks for your feedback!"

		# indicate that the request was a success
		data["success"] = True
	else:
		data["success"] = False

	# return the data dictionary as a JSON response
	return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print("Loading Keras model and Flask starting server... please wait until server has fully started")
	webport = 5000
	app.run(host='0.0.0.0', port=webport, debug=False)

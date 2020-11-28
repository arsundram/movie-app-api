# docs @ http://flask.pocoo.org/docs/1.0/quickstart/
import flask
import pandas as pd
import keras
from keras.models import load_model
import numpy as np
import keras.losses
from tensorflow.keras import backend as K

# instantiate flask 
app = flask.Flask(__name__)

        
def masked_mse(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
    return masked_mse

def masked_rmse_clip(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    y_pred = K.clip(y_pred, 1, 5)
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
    return masked_mse
# we need to redefine our metric function in order 
# to use it when loading the model 
# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#    keras.backend.get_session().run(tf.local_variables_initializer())
#     return auc

# # load the model, and pass in the custom metric function
# global graph
# graph = tf.get_default_graph()
def my_custom_func():
    # your code
    return
model = load_model('mymodel.h5', custom_objects={'masked_mse':masked_mse, 'masked_rmse_clip':masked_rmse_clip})

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])

def predict():
    data = {"success": False}

    params = flask.request.json

    matrix = np.full((1, 3952), 0.0)

    if (params == None):
        params = flask.request.args


    # if parameters are found, return a prediction
    if (params != None):
        if flask.request.headers['Content-Type'] == 'application/json':
            foo = flask.request.get_json()
            for i in foo:
                matrix[0,int(i)] = int(foo[i])
    else:
        matrix[0][0] = 5
        
    predictedMatrix = model.predict(matrix)
    # data["prediction"] = str(predictedMatrix)
    allMovies = pd.read_csv('ml1m_movies.csv',sep='\t', encoding='latin-1', 
                      usecols=['movie_emb_id', 'title','genre'])
    dict = {}
    for pos, row in allMovies.iterrows():
        dict[row['movie_emb_id']] = row['title']+"->"+row['genre']
    pos = 0
    myList = []
    for x in predictedMatrix[0]:  
        try:
            myList.append([x,pos])
            # myList.append([x,dict[pos]])
        except:
            i = 1
        pos = pos+1
    finalList = sorted(myList, key=lambda myList : myList[0], reverse=True)[:50]

    response = []
    for item in finalList:
        response.append(item[1])
    # return a response in json format 
    return flask.jsonify(response)    


@app.route('/')
def test_page():
    message = {'greeting': 'Hello from Flask!'}
    return flask.jsonify(message)  
# start the flask app, allow remote connections 
# app.run(host='0.0.0.0')
# const port = 3000
# server.listen(port,()=>{  // do not add localhost here if you are deploying it
#     console.log("server listening to port "+port);
# })



# @app.route('/add', methods=['GET', 'POST'])
# def add():

#     # POST request
#     if request.method == 'POST':
#         nums = request.get_json()
#         # must return a string value and note nums is a dict of strings
#         print(nums)
#         return str(df1.prediction(float(nums['score'])))

#     # GET request
#     else:
#         message = {'greeting': 'Hello from Flask!'}
#         return jsonify(message)  # serialize and use JSON headers


    # serialize and use JSON headers
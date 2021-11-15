import cv2 as cv
import numpy as np
import tensorflow as tf
import os.path
import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

tf.compat.v1.disable_eager_execution()

def predict(path):

    cap = cv.VideoCapture(path)

    image_size=128
    num_channels=3
    images = []

    data = {
        'label': '',
        'quality': '',
        'probability': ''
    }
    
    width = round(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    newHeight = int(round(height/2))

    graph = tf.Graph()
    graphAQ = tf.Graph()
    graphPQ = tf.Graph()
    graphUQ = tf.Graph()

    default_graph = tf.compat.v1.get_default_graph()

    # Restoring for types model
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph('parkingsurfaceType-model.meta')
    
        # Acessing the graph
        y_pred = graph.get_tensor_by_name("y_pred:0")

        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, len(os.listdir('training_data_type'))))

    sess = tf.compat.v1.Session(graph = graph)
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint('typeCheckpoint/'))


    # Restoring the asphalt quality model
    with graphAQ.as_default():
        saverAQ = tf.compat.v1.train.import_meta_graph('parkingsurfaceAsphaltQuality-model.meta')
        
        # Acessing the graph
        y_predAQ = graphAQ.get_tensor_by_name("y_pred:0")

        xAQ = graphAQ.get_tensor_by_name("x:0")
        y_trueAQ = graphAQ.get_tensor_by_name("y_true:0")
        y_test_imagesAQ = np.zeros((1, len(os.listdir('training_data_asphalt_quality'))))

    sessAQ = tf.compat.v1.Session(graph = graphAQ)
    saverAQ.restore(sessAQ, tf.train.latest_checkpoint('asphaltCheckpoint/'))

    # Restoring the paved quality model 
    with graphPQ.as_default():
        saverPQ = tf.compat.v1.train.import_meta_graph('parkingsurfacePavedQuality-model.meta')
        
        # Acessing the graph
        y_predPQ = graphPQ.get_tensor_by_name("y_pred:0")

        xPQ = graphPQ.get_tensor_by_name("x:0")
        y_truePQ = graphPQ.get_tensor_by_name("y_true:0")
        y_test_imagesPQ = np.zeros((1, len(os.listdir('training_data_paved_quality'))))

    sessPQ = tf.compat.v1.Session(graph = graphPQ)
    saverPQ.restore(sessPQ, tf.compat.v1.train.latest_checkpoint('pavedCheckpoint/'))

    # Restoring unpaved quality model
    with graphUQ.as_default():
        saverUQ = tf.compat.v1.train.import_meta_graph('parkingsurfaceUnpavedQuality-model.meta')
    
        # Acessing the graph
        y_predUQ = graphUQ.get_tensor_by_name("y_pred:0")

        xUQ = graphUQ.get_tensor_by_name("x:0")
        y_trueUQ = graphUQ.get_tensor_by_name("y_true:0")
        y_test_imagesUQ = np.zeros((1, len(os.listdir('training_data_unpaved_quality'))))

    sessUQ = tf.compat.v1.Session(graph = graphUQ)
    saverUQ.restore(sessUQ, tf.compat.v1.train.latest_checkpoint('unpavedCheckpoint/'))


    while cv.waitKey(1) < 0:

        hasFrame, images = cap.read()

        finalimg = images

        if not hasFrame:
            print("Classification done!")
            cv.waitKey(3000)
            break

        images = images[newHeight-5:height-50, 0:width]
        images = cv.resize(images, (image_size, image_size), 0, 0, cv.INTER_LINEAR)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0)

        x_batch = images.reshape(1, image_size, image_size, num_channels)

        y_test_images = np.reshape(images,(-1, 3)) #Reshape
        #
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = sess.run(y_pred, feed_dict=feed_dict_testing)


        outputs = [result[0,0], result[0,1], result[0,2]]

        value = max(outputs)
        index = np.argmax(outputs)


        if index == 0: #Asphalt
            label = 'Asphalt'
            prob = str("{0:.2f}".format(value))
            color = (0, 0, 0)
            x_batchAQ = images.reshape(1, image_size, image_size, num_channels)
            
            y_test_imagesAQ = np.reshape(images,(-1, 3)) #Reshape

            feed_dict_testingAQ = {xAQ: x_batchAQ, y_trueAQ: y_test_imagesAQ}
            resultAQ = sessAQ.run(y_predAQ, feed_dict=feed_dict_testingAQ)
            outputsQ = [resultAQ[0,0], resultAQ[0,1], resultAQ[0,2]]
            valueQ = max(outputsQ)
            indexQ = np.argmax(outputsQ)
            if indexQ == 0: #Asphalt - Standard
                quality = 'Standard'
                colorQ = (0, 255, 0)
                probQ =  str("{0:.2f}".format(valueQ))
            elif indexQ == 1: #Asphalt - Average
                quality = 'Average'
                colorQ = (0, 204, 255)
                probQ =  str("{0:.2f}".format(valueQ))
            elif indexQ == 2: #Asphalt - Bad
                quality = 'Bad'
                colorQ = (0, 0, 255)
                probQ =  str("{0:.2f}".format(valueQ))  
        elif index == 1: #Paved
            label = 'Paved'
            prob = str("{0:.2f}".format(value))
            color = (153, 102, 102)
            x_batchPQ = images.reshape(1, image_size, image_size, num_channels)
            
            y_test_imagesPQ = np.reshape(images,(-1, 3)) #Reshape

            feed_dict_testingPQ = {xPQ: x_batchPQ, y_truePQ: y_test_imagesPQ}
            resultPQ = sessPQ.run(y_predPQ, feed_dict=feed_dict_testingPQ)
            outputsQ = [resultPQ[0,0], resultPQ[0,1], resultPQ[0,2]]
            valueQ = max(outputsQ)
            indexQ = np.argmax(outputsQ)
            if indexQ == 0: #Paved - Standard
                quality = 'Standard'
                colorQ = (0, 255, 0)
                probQ =  str("{0:.2f}".format(valueQ))
            elif indexQ == 1: #Paved - Average
                quality = 'Average'
                colorQ = (0, 204, 255)
                probQ =  str("{0:.2f}".format(valueQ))
            elif indexQ == 2: #Paved - Bad
                quality = 'Bad'
                colorQ = (0, 0, 255)
                probQ =  str("{0:.2f}".format(valueQ))
        elif index == 2: #Unpaved
            label = 'Unpaved'
            prob = str("{0:.2f}".format(value))
            color = (0, 153, 255)
            x_batchUQ = images.reshape(1, image_size, image_size, num_channels)

            y_test_imagesUQ = np.reshape(images,(-1, 2)) #added 4
            #
            feed_dict_testingUQ = {xUQ: x_batchUQ, y_trueUQ: y_test_imagesUQ}
            resultUQ = sessUQ.run(y_predUQ, feed_dict=feed_dict_testingUQ)
            outputsQ = [resultUQ[0,0], resultUQ[0,1]]
            valueQ = max(outputsQ)
            indexQ = np.argmax(outputsQ)
            if indexQ == 0: #Unpaved - Average
                quality = 'Average'
                colorQ = (0, 204, 255)
                probQ =  str("{0:.2f}".format(valueQ))
            elif indexQ == 1: #Unpaved - Bad
                quality = 'Bad'
                colorQ = (0, 0, 255)
                probQ =  str("{0:.2f}".format(valueQ))

        data = {
            'label': label,
            'quality': quality,
            'probability': prob
        }

        sess.close()
        sessAQ.close()
        sessPQ.close()
        sessUQ.close()
        time.sleep(5)

        return data

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route("/mode", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        
        content = request.args['url1']

        if content is None or content == "":
            return jsonify({"error": "no file"})

        try:
            prediction = predict(content)
            print("Test : ", prediction)

            return jsonify(prediction)
        except Exception as e:
            return jsonify({"error 1": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)

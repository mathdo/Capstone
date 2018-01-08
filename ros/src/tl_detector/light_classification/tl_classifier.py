from styx_msgs.msg import TrafficLight
import tensorflow as tf

class TLClassifier(object):

    label_file = '../../../../data/output_labeles.txt'
    graph_file = '../../../../data/output_graph.pb'

    def __init__(self):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(graph_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        self.sess = tf.Session(graph=graph)


    def load_labels():
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())

        return label


    def normalize(img, beta=255):
        cv2.normalize(img, img, alpha=0, beta=beta, norm_type=cv2.NORM_MINMAX)
        

    def prepare_image(file_name):
        img = cv2.imread(file_name)
        img = cv2.resize(img, (input_width, input_height))
    
        normalize(img)
        return img


    def read_tensor_from_image_file(file_name, input_height=224, input_width=224):
        img = prepare_image(file_name)
        
        casted = tf.reshape(img, (-1, input_width, input_height, 3))

        # sess = tf.Session()
        result = self.sess.run(casted)

        return result


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        input_layer = "input"
        output_layer = "final_result"

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name);
        output_operation = graph.get_operation_by_name(output_name);

        t = read_tensor_from_image_file(file_name)

        with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0],
                              {input_operation.outputs[0]: t})

        results = np.squeeze(results)
        
        return results.argmax()


# coding: utf-8

# In[1]:

import nltk
from nltk.stem import  WordNetLemmatizer
wnl = WordNetLemmatizer()


# In[3]:

import numpy as np
import tensorflow as tf
import random
import tflearn
import pandas as pd
import pickle

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS 

class Topic():
    def __init__(self):
        #self.baseDir="C:/Users/A664107/Desktop/test code/abcnn_classify/"
        self.ERROR_THRESHOLD=0.5
        self.sTicketDataFile="/aiap-demo/mockdata.csv"
        self.epoch=1000
        self.dfTicketData=pd.read_csv(self.sTicketDataFile)
        self.topic=[]
        self.solutions=[]

    def trainPath(self):
        print("training Path")
        
	sess = tf.Session()
        
	if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
                print('Usage: mnist_export.py [--training_iteration=x] '
                        '[--model_version=y] export_dir')
                sys.exit(-1)
        if FLAGS.training_iteration <= 0:
                print 'Please specify a positive value for training iteration.'
                sys.exit(-1)
        if FLAGS.model_version <= 0:
                print 'Please specify a positive value for version number.'
                sys.exit(-1)
        
	self.words = []
        self.classes = []
        documents = []
	
	serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
        'query': tf.FixedLenFeature(
            shape=[], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        sentence = tf_example['query']
	
        stopwords = nltk.corpus.stopwords.words("english")
        for index, row in self.dfTicketData.iterrows():
            pattern = str(row['Request Description'])
        # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)
            # add to our words list
            self.words.extend(w)
    # add to documents in our corpus
            documents.append((w, row['Request Category']))
    # add to our classes list
            if row['Request Category'] not in self.classes:
                self.classes.append(row['Request Category'])
# stem and lower each word and remove duplicates
        self.words = [wnl.lemmatize(w.lower()) for w in self.words if w not in stopwords]
        self.words = sorted(list(set(self.words)))
# remove duplicates
        self.classes = sorted(list(set(self.classes)))
        print (len(documents), "documents")
        print (len(self.classes), "classes", self.classes)
        print (len(self.words), "unique stemmed words", self.words)
        
# Start training (apply gradient descent algorithm)

# create a function for this
        #self.model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)

        training = []
# create an empty array for our output
        output_empty = [0] * len(self.classes)
# training set, bag of self.words for each sentence
        for doc in documents:
            # initialize our bag of self.words
            bag = []
            # list of tokenized self.words for the pattern
            patternwords = doc[0]
            # stem each word
            patternwords = [wnl.lemmatize(word.lower()) for word in patternwords]
            # create our bag of self.words array
            for w in self.words:
                bag.append(1) if w in patternwords else bag.append(0)

    		# output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1

            training.append([bag, output_row])
		# shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)
		# create train and test lists
        train_x = list(training[:,0])
        train_y = list(training[:,1])
		# reset underlying graph data
        tf.reset_default_graph()
		# Build neural network
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)
		# Define model and setup tensorboard
        self.model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
        self.model.fit(train_x, train_y, n_epoch=self.epoch, batch_size=8, show_metric=True)
        self.model.save('/aiap-demo/model_Topic.tflearn')
        pickle.dump( {'words':self.words, 'classes':self.classes, 'train_x':train_x, 'train_y':train_y}, open( "/aiap-demo/Path_training_data", "wb" ) )

	# Export model
       
        export_path_base = sys.argv[-1]
        export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))
        print 'Exporting trained model to', export_path
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
	
	indices = tf.nn.top_k(self.classes, 0)
	classify_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
        classes_output_tensor_info = tf.saved_model.utils.build_tensor_info(indices)
        classification_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                        classify_inputs_tensor_info
                },
                outputs={
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                        classes_output_tensor_info
                },
                method_name=tf.saved_model.signature_constants.
                CLASSIFY_METHOD_NAME))

	tensor_info_x = tf.saved_model.utils.build_tensor_info(sentence)
	prediction_signature=(
                tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={'problem':tensor_info_x},
                        outputs={'classes': classes_output_tensor_info},
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                        'predict_query':
                                prediction_signature,
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                classification_signature,
                },
        legacy_init_op=legacy_init_op)
        builder.save()
	print 'Done exporting!'
	
    def reloadModelAndData(self):
        data = pickle.load( open("/aiap-demo/Path_training_data", "rb" ) )
        #data = {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}
        self.words = data['words']
        self.classes = data['classes']
        train_x = data['train_x']
        train_y = data['train_y']
		#getting bag of words
        #p = bow(sentence, words)
        tf.reset_default_graph()
# Build neural network
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)

# Define model and setup tensorboard
        self.model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# load our saved model
        self.model.load('/aiap-demo/model_Topic.tflearn')
        
    def classify(self,sentence):
	    	# generate probabilities from the model
	        results = self.model.predict([self.bow(sentence,self.words)])[0]
	    	# filter out predictions below a threshold
	        results = [[i,r] for i,r in enumerate(results) if r>self.ERROR_THRESHOLD]
	    	# sort by strength of probability
	        results.sort(key=lambda x: x[1], reverse=True)
	        return_list = []
	        for r in results:
	            return_list.append((self.classes[r[0]], r[1])) 
	    	# return tuple of intent and probability
	        return return_list
        

    def clean_up_sentence(self,sentence):
        # tokenize the pattern
        sentencewords = nltk.word_tokenize(sentence)
        # stem each word
        sentencewords = [wnl.lemmatize(word.lower()) for word in sentencewords]
        return sentencewords

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self,sentence, words, show_details=False):
        # tokenize the pattern
        sentencewords = self.clean_up_sentence(sentence)
        # bag of self.words
        bag = [0]*len(self.words)  
        for s in sentencewords:
            for i,w in enumerate(self.words):
                if w == s: 
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)

        return(np.array(bag))

    def response(self, sentence, userID='123', show_details=False):
        topic=self.classify(sentence)
        solutions=[]
        self.solutions=[]
        self.topic=[]
    # if we have a classification then find the matching intent tag
        if topic:
            for index, row in self.dfTicketData.iterrows():
              # find a tag matching the first result
                if row['Request Category'] == topic[0][0]:
                    # a random response from the intent
                    print(row['Solution'])
                    solutions.append(row['Solution'])
                    
        self.solutions=list(set(solutions))
        self.topic=topic

topic = Topic()
tf.app.run(topic.trainPath())
#objTopic=Topic()    
#objTopic.trainPath()            
# '''sentence="""Hi, I hope you are well,

# Please, I'd like your help to create NOLS account with IPM link under:

# The EXTERNAL PROFILE for the following user(s):
# IPM CHANNEL (LAT Subcontractor profile)

# German Ortiz Contreras
# german.ortiz@osctelecoms.com.co
# +573"""     
#print(objTopic.classify(sentence))    '''
    #print(response(sentence))

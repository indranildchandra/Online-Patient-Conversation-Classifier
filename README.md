# Online-Patient-Conversation-Classifier
Online-Patient-Conversation-Classifier can segregate patient conversations from the rest of the group given historically tagged patient data.


# Approach
	1. Embeddings Generation - Using Tensoflow's learn.preprocessing.VocabularyProcessor module to transform each
    word in the text corpus into a vector space (Word2Vec).
    
	2. Patient Conversation Classification - Convolutional Neural Network (CNN) -> uses an embedding layer, followed by a
    convolutional, max-pooling and softmax layer. Used 50% dropout and
    L2 regularization in Cross Entropy loss in to avoid over-fitting.
    
	3. Model Hyperparameters - 
	        Dimensionality of character embedding - 128 ,
	        Filter sizes - (3,4,5) ,
	        Number of filters per filter size - 128 ,
	        Dropout keep probability - 0.5 (50%) ,
	        L2 regularization lambda - 0.0 ,

    	Training Parameters - 
	        Batch Size - 64 ,
	        # of Epochs - 200 ,
	        Evaluate model on Dev Set - every 100 steps ,
	        Save Model checkpoint - every 100 steps ,
	        Number of checkpoints - 5 

    	Dataset Parameters - 
        	Dev sample percentage - 0.1%


# Problem Statement
Identify online patient conversations - Data Science team collaborates with the Social Listening team for automating the process of gaining insights from social media conversations. 
The Social Listening team has to manually validate heart failure related conversations fetched from the social listening tool which scans twitter, Facebook, forums, blogs etc. Such conversations are posted by multiple stakeholders like patients, doctors, media houses, general public, etc. The team needs to identify the patient conversations, so as to dig deeper into them and identify the patient needs. The data science team wants to automate this process by building intelligent algorithms to predict patient conversations. 

Build an Intelligent pipeline that can segregate patient conversations from the rest of the group given historically tagged patient data. You are expected to build an algorithm where they can ingest the social data and get the patient tags - 1 if patient and 0 if not a patient. 

# Dataset
Dataset was provided by Hackerank for a coding contest. Please find the dataset at http://hck.re/pnsNa4. 

### Description of attributes in dataset is given below.
###### Source    -        Type of Social Media Post
###### Host      -        Domain of Social Media Post
###### Link      -        Complete URL of post
###### Date      -        Date of Post
###### Time      -        time Stamp of Post in Eastern Time
###### Time(GMT) -        time Stamp of Post in GMT
###### Title     -        Title of the Post
###### TRANS_CONV_TEXT -  Actual Text Conversation of the Post
###### Patient_Tag   -    Patient Flag (1= Patient, 0=Non-Patient)

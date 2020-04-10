# Disaster Response Pipeline Project

**Introduction**

This is a classification project to match text messages to different categories. Automated dividing of messages might help to take care of an area where a disaster happend due to the fact that the messages can be forwarded to specific organisations that take care for different topics.

**Main Questions**

The main question of this project is the way how data classification works and what benefit we can get out of different approaches.

**Used Libaries**

The following libaries are used in the project:
- re
- pickle
- pandas
- sqlalchemy
- sklearn
- nltk
- plotly
- json

**Architecture**

The project is split in different data files that are in charge of different tasks:

process_data.oy:
	the document reads in the relevant data from csv files, cleans it and saves it as SQL DB

train_classifier:
	the document reads the clean data from a SQL DB and builds a classification model. This model is trained and tested with the data and an evaluation is printed. The trained model is saved as pickle file.
	
run.py:
	this file reads in the data and the trained model and generates some grafical output to visualize the data. Furthermore it is able to grab a query from an html document and classify it with the trainded model.

**Results**

The trained model is able to classify queries and work out the relevant categories. In an grafical interface the assigned categories are highlited.

**CRISP-DM**

_Business_ _understanding_: 
	
	What is the goal?
		- text mesages can be classified automated and fast. This allows a split and a distribution to relevant organisations that are in charge of different tasks

_Data_ _understanding_: 
	
	The data is mainly twofold. Firstly the raw messages are included which contain tons of information (relevant or not). The second part is a classification of every single message to some of 36 categories. This assignment allows us to use the data to train a model.

_Prepare _Data_:

	The data needs to be cleaned on different ways. Firstly, we had to expand the category column from one to 36 columns to be able to work with each category at its own. Secondly, the messages had to be cleaned. In detail we need to remove punctuation, stop words and other not necessary party as endings or same words in differend ways by stemming and lemetazation.
     
_Evaluate_ _the_ _Results_

	We used Gridsearch to fit different key parameters like the maximum number of features or the max number of trees is a random forest algorithm. The evaluation was printed in the output to be able to see the different results on different categories.

**Challenges and Learnings**

	This project took quite a wile for me because I have never worked with categorization before and I had to learn everything new. Overall it was extreamly interesting to see how classification works and to be able to set up my own. I learned a lot of new things and have good understanding of the basics now. In future I plan to deep dive in the different methods of classifications and to implement different methods. Additionally I learned to work with stack overflow and how to post questions there.

**Further Investigarion Necessary At A Later Point In Time**

	I need to claryfy why the pkl file increases in size that much with some more features. My uderstanding is that there is kind of a matching table of keywords and cateories inside. My file increases to some hundered MB when I use more than 1000 features.
	
	Training takes quite long, maybe there is a way to increase the perfoemance.

**Instructions**

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/ or http://0.0.0.0:3001/ 



# Sentiment analysis

A simple sentiment analysis project designed to classify comment on r/starcitizen subreddit

It was tested on Ubunut 18 and Ubunut 20 with python 3.
The training was done on a R5 1600 with 16 Gb of RAM

### Requirements:

* Python 3

### Installation :

* Download the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) and extract it into a aclimdb dataset.
* Install python modules from `requirements.txt` with :
`pip3 install -r requirements.txt` 
Usage of [virtualenv](https://docs.python.org/3/library/venv.html) is heavily recommended
* Download the [word2vec embedding file](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) and put it into the source folder
* To use `reddit_retriever.py` you need to create Oauth credential from a reddit account using a [reddit script app](https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example) and place the results into a credentials.json file (using credentials.json.example as a base)

### Usage:

* Pretrain on aclimdb dataset :
  `python3 train.py --save_path lstm_imdb_vat64.hdf5 --lr 1e-3 --dataset aclimdb_dataset --batch_size 64 --num_hidden 32 --patience 3 --num_cat 2 --use_vat --num_epochs 80 --use_generator --trainable_embedding --save_best_only`
  
* Retrieve comments from reddit and saving tehm into comments_saved.db (will takes several hours):
  `python3 reddit_retriever.py --retrieve_flairs OFFICIAL GAMEPLAY DRAMA FLUFF IMAGE DISCUSSION SOCIAL`
  
* Train on reddit dataset :
 `python3 train.py --save_path lstm_reddit_vat64_pretrained.hdf5 --lr 1e-3 --transfer aclimdb_dataset --dataset reddit_dataset --batch_size 128 --num_hidden 64 --patience 3 --num_cat 3 --use_vat --use_generator --num_epochs 80 --trainable_embedding --load_model lstm_imdb_vat64.hdf5 --save_best_only`
 
* Infering results for all comments :
`python3 infer.py --dataset aclimdb_dataset --load_model lstm_reddit_vat64_pretrained.hdf5 --num_hidden 64 --num_cat 3 --infer_generator --batch_size 256 --save_results results.npz --infer_samples comments_cached.db`

* Doing topic modeling (model will be saved into topic_models folder)
`python3 topic_modeling.py --data comments_cached.db`

* Creating visual analysis (With topic modeling):
`python3 data_analysis.py --saved_results results2.npz --comments_cached comments_cached.db --load_preprocess --topic_model topic_models/model_comments --pos_bias 1`

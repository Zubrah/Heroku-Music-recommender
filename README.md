# Music recommendation system

We all love listening to our favourite music every day. It is pretty hard to find artists or songs similar to our taste and i know for sure ,  we would love a system to do this for us. We have music applications like Spotify that uses content-based and collaborative filtering to recommend us songs similar to what we like.

This code suggests a simple heroku application that shows how the application predicts artists based on the user inputs and return those illustrations!!!!

Hope you habve fun with it.


Link to Video Illustration [Video](https://www.loom.com/share/e6c713ae00514516abe07617dfe1301e).


Link ot the deployed app : [App](https://music-recommendations.herokuapp.com)








## How to install 
Python 3 interpreter is required. It's recommended to use virtual environment for better isolation.

Install requirements:
```bash
pip install -r requirements.txt # alternatively try pip3
```

Setup environment variables (example on Linux):
```bash
export LASTFM_API_KEY=<your API key>
export LASTFM_API_SECRET=<your API secret>
```

To start the application use this command:
```bash
python manage.py runserver
``` 

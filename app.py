import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import requests
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import re
from marvel import Marvel
import itertools
import os
from flask import Flask, request, render_template
m = Marvel("b40f28fccab21ccb6e6b8d6acd617546","ade0b010e4f46812a1ec0eeb8ea2d1fb38c688e4")

global Dict
global node_dict
global rgb_values
global rgb
import matplotlib.pyplot as plt
rgb =pd.DataFrame()
Dict = {}
node_dict = {}

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/', methods =["GET", "POST"])
def gfg():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       comic = request.form.get("comicissue")
       test = map(int, re.findall(r'\d+', comic))
       return rec(test)
    return render_template("form_test.html")

if __name__=='__main__':
   app.run()

class Node:
    def __init__(self, name):
        self.name = name
        self.connection = list()
        return

    def weight(self, node2):
    # FUNCTION: ADDS WEIGHTS TO EACH CONNECTIONS OF CHARACTERS
        Dict[self.name + " & " + node2.name] = 1
        return

    def is_connected(self,node2):
        # FUNCTION: CHEKCS IF TWO CHARACTERS ARE CONNECTED
        if self.connection == None:
            return False
        elif node2.name in self.connection:
            return True
        else:
            return False

    def connect(self, node2):
        # FUNCTION: CONNECTS TWO CHARACTERS IF NOT ALREADY CONNECTED. IF CONNNECTED, CALLS TO CHANGE WEIGHT
        if self.is_connected(node2):
            return self.change_weight(node2)
        else:
            self.connection.append(node2.name)
            node2.connection.append(self.name)
            self.weight(node2)
            return

    def change_weight(self, node2):
        # FUNCTION: CHANGES THE WEIGHT OF CHARACTER CONNECTION
        name = self.name + " & " + node2.name
        v = name in Dict
        if v == False:
            Dict[node2.name + " & " + self.name] +=1
        else:
            Dict[name] +=1
        return v

    def return_weight(self, node2):
        # FUNCTION: RETURNS WEIGHT OF ITEM
        name = self.name + " & " + node2.name
        v = name in Dict
        if v == False:
            return Dict[node2.name + " & " + self.name]
        else:
            return  Dict[name]


def network(df):
        # FUNCTION: CREATES THE NODE NETWORK TO SEE CHARACTER CONNECTION
        # PARAMETER NAMES:
        # DF: DATAFRAME OF THE COMIC BOOK DATA
    for comic in np.arange(len(df["characters"])):
        names = []
        for t in np.arange(df["characters"][comic]['available']):
            names.append(df["characters"][comic]["items"][t]["name"])
            if not df["characters"][comic]["items"][t]["name"] in node_dict:
                node_dict[(df["characters"][comic]["items"][t]["name"])] = Node((df["characters"][comic]["items"][t]["name"]))
            for subset in itertools.combinations(names, 2):
                if node_dict[str(subset[0])].is_connected(node_dict[str(subset[1])]):
                    node_dict[subset[0]].change_weight(node_dict[subset[1]])
                else:
                    node_dict[subset[0]].connect(node_dict[subset[1]])
    return

def get_comic(title, issue, year):
    # FUNCTION: RETRIEVE COMIC BOOK
    # PARAMETER NAMES:
    # LINK: THE LINK OF THE IMAGE
    # NAME: NAME OF THE COMIC BOOK

    return  m.comics.all(limit = 1, format = "comic", titleStartsWith = str(title), startYear = str(year), issueNumber = str(issue), orderBy = "issueNumber")


def get_image(link, name):
        # FUNCTION: RETRIEVE COMIC BOOK IMAGE
        # PARAMETER NAMES:
        # LINK: THE LINK OF THE IMAGE
        # NAME: NAME OF THE COMIC BOOK
    response = requests.get(link)
    name_link = str(name) + ".jpg"
    print(name_link)
    print(name,link)
    save_path = r"C:\Users\huynh\Desktop\Images"
    file_name = name_link
    completeName = os.path.join(save_path, file_name)
    print(completeName)
    file = open(completeName, "wb")
    file.write(response.content)
    file.close()
    return


def create_color2(image, name):
#     FUNCTION: RETURNS THE MOST USED COLOR OF THE GIVEN IMAGE

#     VARIABLES

#     IM: IMAGE
#     WIDTH: WIDTH OF IMAGE
#     HEIGHT: HEIGHT OF IMAGE
#     ARRAY: ALL RGB VALUES OF IMAGE
#     T, NUMBER_OF_ROWS, RANDOM_INDICES, RANDOM_ROWS: VARIABLES USED TO HELP RESHAPE ARRAY INTO TWO DIMENSIONS
#     MEANS: CLUSTERS FROM KMEANS

    im = Image.open(r"C:\Users\huynh\Desktop\Images\\" + image)
    pix = im.load()
    width = im.size[0]
    height = im.size[1]
    array = np.array(im)
    t = array.reshape(-1, array.shape[-1])
    number_of_rows = t.shape[0]
    random_indices = np.random.choice(number_of_rows, size=100, replace=False)
    random_rows = t[random_indices, :]
    kmeans = KMeans(n_clusters=10,random_state = 1)
    test = kmeans.fit(random_rows)
    means = kmeans.cluster_centers_
    means = np.round(means,0).astype(int)
    from collections import Counter, defaultdict
    c = max(Counter(kmeans.labels_), key=Counter(kmeans.labels_).get)
    return means[c]


def get_chracter_comics2(comics):

#     FUNCTION: GETS THE DATA FROM MARVEL'S API AND CREATES THE COLOR PALETTES USING THE PREVIOUS FUNCTIONS. RETURNS THE
#     DATAFRAME AS WELL

#     VARIABLES

#     DF: DATAFRAME OF COMICS FROM MARVEL'S API
#     LINKS: ARRAY OF IMAGE LINKS
#     NAME: NAME OF COMIC BOOK
#     LINK: LINK OF IMAGE FROM SERVER
    df = pd.DataFrame(comics)
    new_df = pd.DataFrame(df["data"][3])[["title", "thumbnail"]]
    links = []
    for x in new_df["thumbnail"]:
        links.append(re.findall("(?P<url>https?://[^\s][^,']+)", str(x)))
    new_df["thumbnail"] = links
    for x in np.arange(len(new_df)):
        name = str(new_df["title"][x])
        link = str(new_df["thumbnail"][x][0]) + ".jpg"
        get_image(link, name)
        t = create_color2(name + ".jpg", name)
    return name, t[0], t[1], t[2]

def color_pal(name, issue, year):
        # FUNCTION: CALLS TO OTHER FUNCTIONS TO GET COLOR PALETTES
    return get_chracter_comics2(get_comic(str(name), str(issue), str(year)))

def calc_dist(df, rgb):
        # FUNCTION: CALCULATES THE CLOSEST COMIC BOOK IMAGE VIA RGB VALUES FROM INPUTTED COMIC BOOK IMAGE
    closest_dist = 1000
    closest_name = ""
    for x in np.arange(len(df["name"])):
        array = np.array([df["red"][x], df["green"][x], df["blue"][x]])
        dist = np.linalg.norm(array-rgb)
        if closest_dist > dist:
            closest_dist = dist
            closest_name = df["name"][x]
    return closest_name

def rec_color(df, colors):
        # FUNCTION: RECOMMEND THE COLORS BY CALLING ALL OTHER FUNCTIONS
    rgb = np.array([colors[0], colors[1], colors[2]])
    rec = calc_dist(df, rgb)
    return rec
def kmeans(df):
        # FUNCTION: USES THE KMEANS ALGORITHEM TO FIND MOST USED RGB VALUE.
    x = df[["red", "green", "blue"]]
    kmeans = KMeans(n_clusters=1,random_state = 1)
    test = kmeans.fit(x)
    means = kmeans.cluster_centers_
    means = np.round(means,0).astype(int)
    from collections import Counter, defaultdict
    c = max(Counter(kmeans.labels_), key=Counter(kmeans.labels_).get)
    return means[c]


# Link to template used https @stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents
def get_key(val):
    for key, value in comic_corpus.items():
        if val == value:
            return key
    print(value)
    return "N/A due to limitations"

def desc(corpus, text):
        # FUNCTION: FINDS THE MOST SIMILAR TEXT
    from sklearn.feature_extraction.text import TfidfVectorizer
    print(text)
    corpus = corpus
    vect = TfidfVectorizer(min_df=1, stop_words="english")
    tfidf = vect.fit_transform(corpus)
    pairwise_similarity = tfidf * tfidf.T
    arr = pairwise_similarity.toarray()
    y = np.fill_diagonal(arr, np.nan)
    input_doc = text
    input_idx = corpus.index(input_doc)
    result_idx = np.nanargmax(arr[input_idx])
    return corpus[result_idx]



dd2019 = pd.read_csv(r"C:\Users\huynh\Desktop\dd2019.csv")


import pickle
import random
with open(r'C:\Users\huynh\Desktop\saved_dictionary.pkl', 'rb') as f:
    comic_corpus = pickle.load(f)

def rec(args):
    user_comics = []
    df_colors = pd.DataFrame(columns = ["red", "green", "blue"])
    corpus = list(comic_corpus.values())
    for x in args:
        user_comics.append(x)
        comics = m.characters.comics(1009262, limit = 100, format = "comic", titleStartsWith = "Daredevil", startYear = "1964", issueNumber = str(x))
        new_df = pd.DataFrame(comics)
        df = pd.DataFrame(new_df["data"][3])
        network(df)
        color_val = color_pal("Daredevil", str(x), "1964")
        rgb_df = pd.DataFrame({"red": [color_val[1]], "green": [color_val[2]], "blue": [color_val[3]]})
        df_colors = df_colors.append(rgb_df, ignore_index = True)

    k = kmeans(df_colors)
    first_rec = max(Dict, key=Dict.get)
    second_rec = rec_color(dd2019, k)
    rc = str(random.choice(user_comics))
    descp = desc(corpus, comic_corpus["Daredevil (1964) #" + rc])
    third_rec = get_key(descp)
    if third_rec == "None":
        rc = str(random.choice(user_comics))
        descp = desc(corpus, comic_corpus["Daredevil (1964) #" + rc])
        third_rec = get_key(descp)
    else:
        third_rec = third_rec
    return "The main chracters in your comics were: " + str(first_rec) + " and we would recommend you: " + str(second_rec) + " based on the colors of your comics and "+ str(third_rec) + " based on your comic themes"

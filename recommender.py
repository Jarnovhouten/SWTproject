import click
import requests
import joblib
import numpy as np
import json
import re

def classify_intent(query):
    # Classify intent:
    # What does user want recommended? Album, artist, songs, ..?
    if 'song' in query:
        return 'song'
    elif 'artist' in query:
        return 'artist'
    elif 'album' in query:
        return 'album'
    else:
        follow_up = click.prompt('Im sorry, did you want a song, artist or album recommendation?\n', type=str)
        classify_intent(follow_up)
    
def match_to_list(query, name_list):
    # Regular expression to match names
    pattern = r'\b(?:' + '|'.join(re.escape(name) for name in name_list) + r')\b'
    
    # convert query to list
    words = query.split()
    # search over increasingly more words, starting from the last word of the string
    for i in range(len(words)):
        index = i+1
        wordlist = words[-index:]
        word_string = ' '.join(wordlist)
        match = re.findall(pattern, word_string, re.IGNORECASE)
        if match != []:
            return match


def find_similar(rec_type, sim_to, number=1):
    # Find similar items based on embeddings
    # rec_type: album / arist / song
    # sim_to: name of album / artist / song to use as reference
    # number: number of items returned
    if rec_type == 'artist':
        sparql_query = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>

            SELECT ?artistURI
            WHERE {{
                {{
                ?artistURI a wsb:Artist_Group ;
                        rdfs:label "{artist}" .
                }}
                UNION
                {{
                ?artistURI a wsb:Artist_Person ;
                        rdfs:label "{artist}" .
                }}
            }}
            """.format(artist=sim_to)
        results = query_sparql_endpoint(sparql_query)
        if results:
            artistURI = results[0]["artistURI"]["value"]
        else:
            print("No results found for the artist.")

    knn_model = joblib.load('embeddings/knn_model.pkl')
    entities = np.load('embeddings/artist_entities.npy')
    embeddings = np.load('embeddings/artist_embeddings_noab.npy')
    query_embedding = embeddings[np.argmax(entities==artistURI)].reshape(1, -1)
    distances, indices = knn_model.kneighbors(query_embedding, n_neighbors=number)
    for i in indices:
        print(entities[i])


def get_recommendations(user_query):
    intent = classify_intent(user_query)

    # logic for recommendations
    if intent == 'album':
        sparql_query = """
        PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>
        PREFIX dcterms: <http://purl.org/dc/terms/>

        SELECT DISTINCT ?title
        WHERE {
        ?album a wsb:Album ;
                dcterms:title ?title .
        }
        ORDER BY RAND()
        LIMIT 10
        """
        results = query_sparql_endpoint(sparql_query)
        print("here are 10 random albums:")
        for title_obj in results:
            title = title_obj['title']['value']
            print(title)
    elif intent == 'artist':
        sparql_query = """
        PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>
        PREFIX foat: <http://xmlns.com/foaf/0.1/>
        
        SELECT DISTINCT ?Name
        WHERE {
        {?Artist a wsb:Artist_Person ; foaf:name ?Name . }
        UNION
        {?Artist a wsb:Artist_Group ; foaf:name ?Name . }
        }
        ORDER BY RAND()
        LIMIT 10
        """
        results = query_sparql_endpoint(sparql_query)
        print("here are 10 random artists:")
        for name_obj in results:
            name = name_obj['Name']['value']
            print(name)
    elif intent == 'song':
        # example sparql query to get 10 random song titles from wasabi kg
        sparql_query = """
        PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>
        PREFIX dcterms: <http://purl.org/dc/terms/>

        SELECT DISTINCT ?title
        WHERE {
        ?song a wsb:Song ;
                dcterms:title ?title .
        }
        ORDER BY RAND()
        LIMIT 10
            """
        results = query_sparql_endpoint(sparql_query)
        print("here are 10 random songs:")
        for title_obj in results:
            title = title_obj['title']['value']
            print(title)


def query_sparql_endpoint(query):
    # Send sparql query to sparql endpoint and return results
    url = 'http://wasabi.inria.fr/sparql'
    results = requests.get(url, params={'query': query, 'format': 'json'}).json()
    return results['results']['bindings']

if __name__ == '__main__':
    query = click.prompt('Hi! How can I help?\n', type=str)
    intent = classify_intent(query)
    if intent == "artist":
        with open('artistnames.json', 'r') as json_file: 
            artist_list = json.load(json_file)
        artist = match_to_list(query, artist_list)
    elif intent == "album":
        with open('albumtitles.json', 'r') as json_file: 
            album_list = json.load(json_file)
        album = match_to_list(query, album_list)
    elif intent == "song":
        with open('songtitles.json', 'r') as json_file: 
            song_list = json.load(json_file)
        song = match_to_list(query, song_list)

    
    
    # get_recommendations(query)

    # print()
    
    #find_similar('artist', 'Ed Sheeran', 3)

    # print()
    
    find_similar('artist', 'Ed Sheeran', 3)

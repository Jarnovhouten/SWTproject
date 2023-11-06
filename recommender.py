import click
import requests
import joblib
import numpy as np
import json
import re
from text_to_num import alpha2digit

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
    # search over increasingly more words starting from the last word of the list
    for i in range(len(words)):
        index = i+1
        wordlist = words[-index:]
        word_string = ' '.join(wordlist)
        match = re.findall(pattern, word_string, re.IGNORECASE)
        # Stop searching once a match is found
        if match != []:
            return match

def get_number(query):
    # Convert textual representations of numbers into digits
    digitsquery = alpha2digit(query, lang="en")
    # Use regular expression to find all integer-like patterns in the query
    numbers = [int(match) for match in re.findall(r'\d+', digitsquery)]
    
    if numbers:
        return numbers[0]  # Return the first found number
    else:
        return 3 # Return 3 by default

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

    knn_model = joblib.load('embeddings/artist_knn_model.pkl')
    entities = np.load('embeddings/artist_entities.npy')
    embeddings = np.load('embeddings/artist_embeddings_noab.npy')
    query_embedding = embeddings[np.argmax(entities==artistURI)].reshape(1, -1)
    distances, indices = knn_model.kneighbors(query_embedding, n_neighbors=number+1)
    return [entities[i] for i in indices][0][1:]

def dict_to_sparql(input_dict):
    sparql_str = ""
    for key, value in input_dict.items():
        sparql_str += f"{key} '{value}' ; "
    sparql_str = sparql_str[:-2] + " ."  
    return sparql_str

def SPARQL_builder(query_type, filters):
    # query_type: 'artist', 'album' or 'song'
    # filters: dict of filters, e.g. {'schema:genre': 'Pop', 'dcterms:language': 'eng'}
    prefixes = """
        PREFIX mo: <http://purl.org/ontology/mo/>
        PREFIX schema: <http://schema.org/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        """
    if query_type == 'artist':
        query = prefixes + """
            SELECT DISTINCT ?Name 
            WHERE {
            {
            ?artist a wsb:Artist_Person;  foaf:name ?Name ;
            """
        query += dict_to_sparql(filters)
        query += '}  UNION { ?artist a wsb:Artist_Group; foaf:name ?Name ;'
        query += dict_to_sparql(filters)
        query += '} } ORDER BY RAND() LIMIT 3'
        return query


def get_recommendations(user_query):
    intent = classify_intent(user_query)

    if intent == "artist":
        with open('embeddings/Name dictionaries/artistnames.json', 'r') as json_file: 
            artist_list = json.load(json_file)
        artist = match_to_list(query, artist_list)
        if artist:
            similar = find_similar('artist', artist[0], 3)
            print('Here are 3 artists similar to {}:'.format(artist[0]))
            for artist in similar:
                print(artist)
        else: 
            filters = dict()
            # Check if there are genres in the query and if so add them to filters dict
            with open('embeddings/Name dictionaries/genres.json', 'r') as json_file: 
                genre_list = json.load(json_file)
            genre = match_to_list(query, genre_list)
            filters['schema:genre'] = genre[0]

            # Check for other filters

            # Build sparql query and get results
            results = query_sparql_endpoint(SPARQL_builder('artist', filters))
            for name_obj in results:
                name = name_obj['Name']['value']
                print(name)
    elif intent == "album":
        with open('embeddings/Name dictionaries/albumtitles.json', 'r') as json_file: 
            album_list = json.load(json_file)
        album = match_to_list(query, album_list)
    elif intent == "song":
        with open('embeddings/Name dictionaries/songtitles.json', 'r') as json_file: 
            song_list = json.load(json_file)
        song = match_to_list(query, song_list)

    # logic for recommendations
    # if intent == 'album':
    #     sparql_query = """
    #     PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>
    #     PREFIX dcterms: <http://purl.org/dc/terms/>

    #     SELECT DISTINCT ?title
    #     WHERE {
    #     ?album a wsb:Album ;
    #             dcterms:title ?title .
    #     }
    #     ORDER BY RAND()
    #     LIMIT 10
    #     """
    #     results = query_sparql_endpoint(sparql_query)
    #     print("here are 10 random albums:")
    #     for title_obj in results:
    #         title = title_obj['title']['value']
    #         print(title)
    # elif intent == 'artist':
    #     sparql_query = """
    #     PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>
    #     PREFIX foat: <http://xmlns.com/foaf/0.1/>
        
    #     SELECT DISTINCT ?Name
    #     WHERE {
    #     {?Artist a wsb:Artist_Person ; foaf:name ?Name . }
    #     UNION
    #     {?Artist a wsb:Artist_Group ; foaf:name ?Name . }
    #     }
    #     ORDER BY RAND()
    #     LIMIT 10
    #     """
    #     results = query_sparql_endpoint(sparql_query)
    #     print("here are 10 random artists:")
    #     for name_obj in results:
    #         name = name_obj['Name']['value']
    #         print(name)
    # elif intent == 'song':
    #     # example sparql query to get 10 random song titles from wasabi kg
    #     sparql_query = """
    #     PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>
    #     PREFIX dcterms: <http://purl.org/dc/terms/>

    #     SELECT DISTINCT ?title
    #     WHERE {
    #     ?song a wsb:Song ;
    #             dcterms:title ?title .
    #     }
    #     ORDER BY RAND()
    #     LIMIT 10
    #         """
    #     results = query_sparql_endpoint(sparql_query)
    #     print("here are 10 random songs:")
    #     for title_obj in results:
    #         title = title_obj['title']['value']
    #         print(title)


def query_sparql_endpoint(query):
    # Send sparql query to sparql endpoint and return results
    url = 'http://wasabi.inria.fr/sparql'
    results = requests.get(url, params={'query': query, 'format': 'json'}).json()
    return results['results']['bindings']

if __name__ == '__main__':
    query = click.prompt('Hi! How can I help?\n', type=str)

    get_recommendations(query)

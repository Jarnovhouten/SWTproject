import click
import requests
import joblib
import numpy as np
import json
import re
import spacy
from text_to_num import alpha2digit


def classify_intent(query):
    """Returns the intent of a given user query.

    Argument:
    query (string): User query

    Returns:
    string: 'album', 'artist' or 'song'
    If no intent is found, user is asked to rephrase the request.
    """

    if any(keyword in query for keyword in ['song', 'tune', 'track',
                                            'ballad', 'composition']):
        return 'song'
    elif any(keyword in query for keyword in ['artist', 'performer',
                                              'musician', 'singer',
                                              'anyone', 'anybody',
                                              'someone', 'somebody']):
        return 'artist'
    elif any(keyword in query for keyword in ['album', 'record', 'music']):
        return 'album'
    else:
        follow_up = click.prompt(
            "Sorry, I didn't understand. Please rephrase your request.\n",
            type=str,
            prompt_suffix='>')
        classify_intent(follow_up)


def match_to_list(query, name_list):
    """Checks if an item from the list is found in the query and
    returns this item if so.

    Argument:
    query (string): User query
    name_list (list): List of items to match

    Returns:
    string: Matched item from given list
    Returns None if there is no match
    """
    
    # Convert name_list to lowercase for case-insensitive matching
    lowercase_name_list = [name.lower() for name in name_list]
    # Regular expression to match names
    pattern = r'\b(?:' + '|'.join(re.escape(name) for name in lowercase_name_list) + r')\b'
    
    # convert query to lowercase
    lowercase_query = query.lower()

    #Initialize list of matches
    matches = []
    # find matches
    matches_lower = re.findall(pattern, lowercase_query)
    if matches_lower:
        # convert lowercase matches back to proper name via list indexes
        for i in range(len(matches_lower)):
            matched_index = lowercase_name_list.index(matches_lower[i])
            matches.append(name_list[matched_index])
    
    #make matches consist of only unique items
    matches = list(set(matches))
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        #assume the longest matching string is the target
        match = max(matches, key=len)
        return match
    else:
        return None


def get_number(query):
    """Takes in a query string and returns any numbers found in the query
    (both as digits or text) as an integer.

    Argument:
    query (string): User query

    Returns:
    integer: The first number found in the given query
    Returns None if no number is found
    """
    # Convert textual representations of numbers into digits
    digitsquery = alpha2digit(query, lang="en")
    # Use regular expression to find all integer-like patterns in the query
    numbers = [int(match) for match in re.findall(r'\d+', digitsquery)]

    if numbers:
        return numbers[0]  # Return the first found number


def get_location(query):
    """Takes in a query string and returns any locations such as country
    found in the string

    Argument:
    query (string): User query

    Returns:
    string: Location found in string
    Returns None if no location is found"""
    # Returns the first location found in a query
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(query)
    locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    if locations:
        return locations[0]


def find_similar_artist(artist, number=1, return_uri=False):
    """Finds similar artists based on given artist using embeddings.

    Arguments:
    artist (string): Name of artist that similar artists should be based on.
    number (int): Number of similar artist that should be returned.
    return_uri (bool): If set to True, a list of URIs is returned.

    Returns:
    if return_uri=False:
        list: List of artist names.
    if return_uri=True:
        list: List of artist URIs."""
    # Find artist URI to find embedding
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
        """.format(artist=artist)
    results = query_sparql_endpoint(sparql_query)
    if results:
        artistURI = results[0]["artistURI"]["value"]
        # Load in trained knn model, entities and embeddings
        knn_model = joblib.load('embeddings/artist_knn_model.pkl')
        entities = np.load('embeddings/artist_entities.npy')
        embeddings = np.load('embeddings/artist_embeddings.npy')
        # Find embedding of given artist and use this to find similar
        # artist uris with knn model
        query_embedding = (
            embeddings[np.argmax(entities == artistURI)]
            .reshape(1, -1)
        )
        _, indices = knn_model.kneighbors(query_embedding,
                                          n_neighbors=number+1)
        sim_uris = [entities[i] for i in indices][0][1:]
        if not return_uri:
            # Turn uris back into artist names through sparql query
            sim_artists = []
            for uri in sim_uris:
                sparql_query = """
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                    SELECT ?name
                    WHERE {{
                    <{artistURI}> rdfs:label ?name .
                        }}
                    """.format(artistURI=uri)
                artist = query_sparql_endpoint(sparql_query)
                sim_artists.append(artist[0]['name']['value'])
            return sim_artists
        else:
            return sim_uris
    else:
        print("Could not find similar artists.")


def find_similar_album(album, number=1, return_uri=False):
    """Finds similar albums based on given album using embeddings.

    Arguments:
    album (string): Title of album that similar albums should be based on.
    number (int): Number of similar albums that should be returned.
    return_uri (bool): If set to True, a list of URIs is returned.

    Returns:
    if return_uri=False:
        list: List of items in the form of: artist name - album title.
    if return_uri=True:
        list: List of album URIs."""
    # Find album URI to find embedding
    sparql_query = """
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>

        SELECT DISTINCT ?albumURI
        WHERE {{
        ?albumURI a wsb:Album ;
                    dcterms:title "{album}" .
        }}

        """.format(album=album)
    results = query_sparql_endpoint(sparql_query)
    if results:
        albumURI = results[0]["albumURI"]["value"]
        # Load in trained knn model, entities and embeddings
        knn_model = joblib.load('embeddings/album_knn_model.pkl')
        entities = np.load('embeddings/album_entities.npy')
        embeddings = np.load('embeddings/album_embeddings.npy')
        # Find embedding of given artist and use this to find similar
        # artist uris with knn model
        query_embedding = (
            embeddings[np.argmax(entities == albumURI)]
            .reshape(1, -1)
        )
        _, indices = knn_model.kneighbors(query_embedding,
                                          n_neighbors=number+1)
        sim_uris = [entities[i] for i in indices][0][1:]
        # Turn uris back into artist names through sparql query
        sim_albums = []
        if not return_uri:
            for uri in sim_uris:
                sparql_query = """
                PREFIX dcterms:  <http://purl.org/dc/terms/>
                PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>
                PREFIX mo: <http://purl.org/ontology/mo/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                SELECT ?title ?artist
                WHERE {{
                <{albumURI}> a wsb:Album;
                    dcterms:title ?title ;
                    mo:performer ?performer .
                ?performer rdfs:label ?artist .
                }}""".format(albumURI=uri)
                album = query_sparql_endpoint(sparql_query)
                sim_albums.append(album[0]['artist']['value'] +
                                  ' - ' + album[0]['title']['value'])
            return sim_albums
        else:
            return sim_uris
    else:
        print("Could not find similar albums.")


def find_similar_song(song, number=1):
    """Finds similar songs based on the album the song is featured on.
    If song is not featured on an album, recommendation will be based on
    the artist that performed the song.

    Arguments:
    song (string): Title of song that similar songs should be based on.
    number (int): Number of similar songs that should be returned.

    Returns:
    list: List of items in the form of: artist name - song title."""
    # Find title of the album the songs is featured on.
    sparql_query = """
    PREFIX dcterms:  <http://purl.org/dc/terms/>
    PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>
    PREFIX mo: <http://purl.org/ontology/mo/>
    PREFIX schema: <http://schema.org/>

    SELECT DISTINCT ?album
    WHERE {{
        ?songURI a wsb:Song;
            wsb:title_without_accent "{song}";
            schema:album ?albumURI.
        ?albumURI dcterms:title ?album .
    }}
    LIMIT 1""".format(song=song)
    album = query_sparql_endpoint(sparql_query)[0]['album']['value']
    # If an album is found, find a song from each similar album
    if album:
        album_uris = find_similar_album(album=album,
                                        number=number,
                                        return_uri=True)
        similar_songs = []
        for uri in album_uris:
            sparql_query = """
                PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>
                PREFIX schema: <http://schema.org/>
                PREFIX mo: <http://purl.org/ontology/mo/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX dcterms: <http://purl.org/dc/terms/>

                SELECT DISTINCT ?song ?artist
                WHERE {{
                    ?songURI a wsb:Song;
                        schema:album <{uri}> ;
                        wsb:title_without_accent ?song .
                    <{uri}> mo:performer ?performerURI .
                    ?performerURI rdfs:label ?artist .
                }}
                ORDER BY RAND()
                LIMIT 1""".format(uri=uri)
            results = query_sparql_endpoint(sparql_query)
            if results:
                similar_songs.append(results[0]['artist']['value'] +
                                     ' - ' + results[0]['song']['value'])
        return similar_songs
    else:
        # If no album was found, try to find performer of the song
        sparql_query = """
        PREFIX dcterms:  <http://purl.org/dc/terms/>
        PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>
        PREFIX mo: <http://purl.org/ontology/mo/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?performer
        WHERE {{
            ?songURI a wsb:Song;
                wsb:title_without_accent "{song}";
                mo:performer ?performerURI.
            ?performerURI rdfs:label ?performer .
        }}
        LIMIT 1""".format(song=song)
        performer = query_sparql_endpoint(sparql_query)[0]['performer']['value']
        # If a performer is found find a song from each similar performer
        if performer:
            performer_uris = find_similar_artist(artist=performer,
                                                 number=number,
                                                 return_uri=True)
            similar_songs = []
            for uri in performer_uris:
                sparql_query = """
                    PREFIX wsb: <http://ns.inria.fr/wasabi/ontology/>
                    PREFIX mo: <http://purl.org/ontology/mo/>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                    SELECT DISTINCT ?song ?artist
                    WHERE {{
                        ?songURI a wsb:Song;
                            mo:performer <{uri}> ;
                            wsb:title_without_accent ?song .
                        <{uri}> rdfs:label ?artist .
                    }}
                    ORDER BY RAND()
                    LIMIT 1""".format(uri=uri)
                results = query_sparql_endpoint(sparql_query)
                if results:
                    similar_songs.append(results[0]['artist']['value'] +
                                         ' - ' + results[0]['song']['value'])
            return similar_songs
        else:
            print('Could not find similar songs.')


def list_to_sparql(input_list):
    """Turns a list of items into sparql format. Adds ';' in between items
    and '.' after the last item.

    Argument:
    input_list (list): List of items in the form of: "predicate 'value'"
    e.g. ["schema:genre'Pop'", "dcterms:language 'eng'"]

    Returns:
    string: Items from list formatted to be used in a sparql query"""

    sparql_str = ""
    for item in input_list:
        sparql_str += f"{item} ; "
    sparql_str = sparql_str[:-2] + " ."
    return sparql_str


def SPARQL_builder(query_type, filters, number):
    """Creates sparql queries based on query type and filters

    Arguments:
    query_type (string): Possible values are 'artist', 'album' or 'song'
    filters (list): List of filters,
    e.g. ["schema:genre'Pop'", "dcterms:language 'eng'"]
    number (int): Number of results to limit the query to

    Returns:
    string: Sparql query"""

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
        query += list_to_sparql(filters)
        query += '}  UNION { ?artist a wsb:Artist_Group; foaf:name ?Name ;'
        query += list_to_sparql(filters)
        query += '} } ORDER BY RAND() LIMIT '
        query += str(number)
        return query
    elif query_type == 'album':
        query = prefixes + """
        SELECT DISTINCT ?title ?artist
        WHERE {
        ?album a wsb:Album ;
            dcterms:title ?title ;
            mo:performer ?artistURI ;
        """
        query += list_to_sparql(filters)
        query += '?artistURI rdfs:label ?artist .'
        query += '} ORDER BY RAND() LIMIT '
        query += str(number)
        return query
    elif query_type == 'song':
        query = prefixes + """
        SELECT DISTINCT ?title ?artist
        WHERE {
        ?song a wsb:Song;
            wsb:title_without_accent ?title ;
            mo:performer ?artistURI ;
        """
        query += list_to_sparql(filters)
        query += '?artistURI rdfs:label ?artist. } ORDER BY RAND() LIMIT '
        query += str(number)
        return query


def get_recommendations(user_query):
    """Prints out song, artist or album recommendations based on a given query

    Argument:
    user_query (string): User query

    Returns: None
    Prints recommendations. One recommendation per line."""

    intent = classify_intent(user_query)
    number = get_number(user_query)
    if not number:
        number = 3

    if intent == "artist":
        with open('embeddings/Name dictionaries/artistnames.json',
                  'r') as json_file:
            artist_list = json.load(json_file)
        artist = match_to_list(query, artist_list)
        if artist:
            similar = find_similar_artist(artist, number)
            print('Here are {} artists similar to {}:'.format(number, artist))
            for artist in similar:
                print('-', artist)
        else:
            filters = []
            # Check if there are genres in the query and if so add them
            # to filters dict
            with open('embeddings/Name dictionaries/genres.json',
                      'r') as json_file:
                genre_list = json.load(json_file)
            genre = match_to_list(query, genre_list)

            if genre:
                filters.append("schema:genre '{}'".format(genre))

            # Check for location
            location = get_location(query)
            if location:
                filters.append('wsb:location [ wsb:country "{}" ]'.format(
                               location))

            # Build sparql query and get results
            if filters:
                results = query_sparql_endpoint(SPARQL_builder('artist',
                                                               filters,
                                                               number))
                for i, name_obj in enumerate(results):
                    if i <= number:
                        name = name_obj['Name']['value']
                        print('-', name)

    elif intent == "album":
        with open('embeddings/Name dictionaries/albumtitles.json',
                  'r') as json_file:
            album_list = json.load(json_file)
        album = match_to_list(query, album_list)
        if album:
            similar = find_similar_album(album, number)
            print('Here are {} albums similar to {}:'.format(number, album))
            for album in similar:
                print(album)
        else:
            filters = []
            # Check if there are genres in the query and if so add them
            # to filters dict
            with open('embeddings/Name dictionaries/genres.json',
                      'r') as json_file:
                genre_list = json.load(json_file)
            genre = match_to_list(query, genre_list)

            if genre:
                filters.append("mo:genre '{}'".format(genre))

            # Build sparql query and get results
            if filters:
                results = query_sparql_endpoint(SPARQL_builder('album',
                                                               filters,
                                                               number))
                for i, name_obj in enumerate(results):
                    if i <= number:
                        artist = name_obj['artist']['value']
                        title = name_obj['title']['value']
                        print('-', artist, '-', title)

    elif intent == "song":
        with open('embeddings/Name dictionaries/songtitles.json',
                  'r') as json_file:
            song_list = json.load(json_file)
        song = match_to_list(query, song_list)
        if song:
            similar = find_similar_song(song, number)
            print('Here are {} similar song(s) to {}:'.format(number, song))
            for song in similar:
                print(song)
        else:
            filters = []
            # Check if there are genres in the query and if so add them
            # to filters dict
            # Make sure genre is added last in the dict for songs
            with open('embeddings/Name dictionaries/genres.json',
                      'r') as json_file:
                genre_list = json.load(json_file)
            genre = match_to_list(query, genre_list)

            if genre:
                filters.append("schema:album ?album. ?album mo:genre '{}'"
                               .format(genre))

            # Build sparql query and get results
            if filters:
                results = query_sparql_endpoint(SPARQL_builder('song',
                                                               filters,
                                                               number))
                for i, name_obj in enumerate(results):
                    if i <= number:
                        artist = name_obj['artist']['value']
                        title = name_obj['title']['value']
                        print('-', artist, '-', title)


def query_sparql_endpoint(query):
    """Sends sparql query to sparql endpoint and returns results

    Argument:
    query (string): User query

    Returns:
    list of dict: A list of dictionaries with the results in JSON format.
    A dictionary contains variables returned by the sparql endbpoint as keys
    and the results as values.
    """
    url = 'http://wasabi.inria.fr/sparql'
    results = requests.get(url,
                           params={'query': query, 'format': 'json'}).json()
    return results['results']['bindings']


if __name__ == '__main__':
    query = click.prompt('Hi! How can I help?\n', type=str, prompt_suffix='>')
    get_recommendations(query)

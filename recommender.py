import click
import requests

def classify_intent(query):
    # Classify intent:
    # What does user want recommended? Album, artist, songs, ..?
    return 'song'

def find_similar(sim_to, number=1):
    # Find similar items based on embeddings
    # sim_to: album / arist / song
    # number: number of items returned
    pass

def get_recommendations(user_query):
    intent = classify_intent(user_query)

    # logic for recommendations
    if intent == 'album':
        pass
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
    # right now, whatever you type in will get you 10 random songs
    get_recommendations(query)
    

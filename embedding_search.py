from karateclub.dataset import GraphReader
from karateclub import Diff2Vec
import networkx as nx
from sklearn.neighbors import KDTree
from rdflib import Graph
import requests

def create_graph(rdf_url):
    # Fetch RDF data from the URL
    response = requests.get(rdf_url)

    if response.status_code == 200:
        rdf_data = response.text

        # Parse RDF data
        g = Graph()
        g.parse(data=rdf_data, format="turtle")  # Assuming the data is in Turtle format

        # Create a NetworkX graph
        rdf_graph = nx.Graph()

        # Iterate through RDF triples and add nodes and edges to the NetworkX graph
        for subj, pred, obj in g:
            rdf_graph.add_node(subj)
            rdf_graph.add_node(obj)
            rdf_graph.add_edge(subj, obj, predicate=pred)
        return rdf_graph 
    else:
        print("Failed to fetch RDF data from the URL.")
        exit()
   
    

    
if __name__ == "__main__":
    #Read in graph, should be our own graph fitted to this function in the future
    rdf_url = "http://ns.inria.fr/wasabi/ontology"
    reader = GraphReader(create_graph(rdf_url))
    graph = reader.get_graph()
    #Construct model
    model=Diff2Vec(diffusion_number=2, diffusion_cover=20, dimensions=16)
    #fit model on the KG
    model.fit(graph)
    #Fetch embeddings from the fitted model
    X = model.get_embedding()
    #Create a KDTree for nearest neighbor search
    tree = KDTree(X, leaf_size=10)
    #Test the KDTree to find the 3 nearest neighbors to the first item in the embeddings
    distance, indices = tree.query(X[0:1], k=3)
    print("Indices of nearest neighbors: {}".format(indices))
    print("Distances to nearest neighbors: {}".format(distance))


    
    
    
    
    # Print the attributes of all nodes
    #for node_id, attributes in graph.nodes(data=True):
        #print(f"Node {node_id}: Attributes - {attributes}")


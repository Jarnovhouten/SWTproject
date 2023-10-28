from karateclub.dataset import GraphReader
from karateclub import Diff2Vec
import networkx as nx
from sklearn.neighbors import KDTree


if __name__ == "__main__":
    #Read in graph, should be our own graph fitted to this function in the future
    reader = GraphReader("deezer")
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


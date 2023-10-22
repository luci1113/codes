import networkx as nx

# Create a directed graph representing web pages and links
G = nx.DiGraph()
G.add_edges_from([
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'A'),
    ('C', 'B'),
    ('D', 'C'),
    ('E', 'B'),
])

# Calculate PageRank
pagerank = nx.pagerank(G)

# Display PageRank scores for each web page
for page, score in pagerank.items():
    print(f'Page: {page}, PageRank: {score:.3f}')

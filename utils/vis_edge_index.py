import numpy as np
import matplotlib.pyplot as plt
import torchfrom itertools import permutations

# Assuming you already have edge_index defined
num_nodes = 10
edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()

# Convert edge_index to NumPy array
edge_index_np = edge_index.numpy()

# Extract source and target nodes from edge_index
source_nodes = edge_index_np[0]
target_nodes = edge_index_np[1]

# Create a plot using Matplotlib
plt.figure(figsize=(6, 4))
plt.scatter(source_nodes, target_nodes, marker='o', color='black')
plt.title("Graph Edges")
plt.xlabel("Source Nodes")
plt.ylabel("Target Nodes")
plt.xlim(0, num_nodes - 1)
plt.ylim(0, num_nodes - 1)
plt.grid(True)
plt.show()

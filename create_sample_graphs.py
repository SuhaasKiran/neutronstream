import dgl
import torch
import numpy as np

# TODO change graph implementation with respect to the training code and training input graph
# base graph is the graph at time t=0
# events graph refers to the graph of events after for time t>0

# create base graph 1
base_graph_num_edges = 14
base_graph_num_nodes = 12
base_graph_time = 0.0
base_src_nodes = torch.tensor([9,8,9,6,7,2,7,3,6,1,4,1,0,5])
base_dst_nodes = torch.tensor([0,9,8,7,6,7,2,6,3,3,3,5,5,0])
    
g = dgl.graph((base_src_nodes, base_dst_nodes), num_nodes=base_graph_num_nodes)
    
base_edge_feats = torch.randn(base_src_nodes.shape[0], 100, dtype=torch.float32)  # Random features with shape (172,)
base_edge_labels = torch.bernoulli(torch.empty(base_src_nodes.shape[0], dtype=torch.float64).uniform_(0, 1))
base_edge_timestamps = torch.zeros(base_src_nodes.shape[0], dtype=torch.float64)
    
g.ndata['_ID'] = torch.ones(g.num_nodes(), 3)  
g.edata['feats'] = base_edge_feats 
g.edata['label'] = base_edge_labels
g.edata['timestamp'] = base_edge_timestamps
data = g

dgl.save_graphs("base_graph_1.bin", data)
print("Graph saved to 'base_graph_1.bin'")

# create base graph 2  
base_graph_num_nodes = 10
base_graph_time = 0.0
base_src_nodes = torch.tensor([0,1,2,3,2,5,5,5,6,4])
base_dst_nodes = torch.tensor([3,2,3,2,5,2,4,6,8,6])
    
g = dgl.graph((base_src_nodes, base_dst_nodes), num_nodes=base_graph_num_nodes)
    
base_edge_feats = torch.randn(base_src_nodes.shape[0], 100, dtype=torch.float32)  # Random features with shape (172,)
# edge_labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float64)  # Example labels
base_edge_labels = torch.bernoulli(torch.empty(base_src_nodes.shape[0], dtype=torch.float64).uniform_(0, 1))
base_edge_timestamps = torch.zeros(base_src_nodes.shape[0], dtype=torch.float64)

g.ndata['_ID'] = torch.ones(g.num_nodes(), 3)  
g.edata['feats'] = base_edge_feats 
g.edata['label'] = base_edge_labels
g.edata['timestamp'] = base_edge_timestamps
data = g

dgl.save_graphs("base_graph_2.bin", data)
print("Graph saved to 'base_graph_2.bin'")
    
# create the events graph 1
events_src_nodes = torch.tensor([5,6,1,7,0,2,10,11])  # Source nodes
events_dst_nodes = torch.tensor([4,10,4,8,1,6,7,9])  # Destination nodes
events_edge_feats = torch.randn(events_src_nodes.shape[0], 100, dtype=torch.float32)  # Random features with shape (100,)
# edge_labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float64)  # Example labels
events_edge_labels = torch.bernoulli(torch.empty(events_src_nodes.shape[0], dtype=torch.float64).uniform_(0, 1))
events_edge_timestamps = torch.tensor(list(np.arange(1.0, 9.0, 1.0)), dtype=torch.float64)  # Example timestamps (UNIX time)
events_g = dgl.graph((events_src_nodes, events_dst_nodes))
events_g.ndata['_ID'] = torch.ones(events_g.num_nodes(), 3)  
events_g.edata['feats'] = events_edge_feats 
events_g.edata['label'] = events_edge_labels
events_g.edata['timestamp'] = events_edge_timestamps

dgl.save_graphs("event_graph_1.bin", events_g)
print("Graph saved to 'event_graph_1.bin'")

events_src_nodes = torch.tensor([0,4,1,7,9])  # Source nodes
events_dst_nodes = torch.tensor([1,7,3,6,8])  # Destination nodes
events_edge_feats = torch.randn(events_src_nodes.shape[0], 100, dtype=torch.float32)  # Random features with shape (100,)
# edge_labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float64)  # Example labels
events_edge_labels = torch.bernoulli(torch.empty(events_src_nodes.shape[0], dtype=torch.float64).uniform_(0, 1))
events_edge_timestamps = torch.tensor(list(np.arange(1.0, float(events_src_nodes.shape[0]+1), 1.0)), dtype=torch.float64)  # Example timestamps (UNIX time)
events_g = dgl.graph((events_src_nodes, events_dst_nodes))
events_g.ndata['_ID'] = torch.ones(events_g.num_nodes(), 3)  
events_g.edata['feats'] = events_edge_feats 
events_g.edata['label'] = events_edge_labels
events_g.edata['timestamp'] = events_edge_timestamps

dgl.save_graphs("event_graph_2.bin", events_g)
print("Graph saved to 'event_graph_2.bin'")
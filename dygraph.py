import dgl
import torch
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, required=True, help="Path to the base graph file")
parser.add_argument("--event", type=str, required=True, help="Path to the event graph file")
args = parser.parse_args()

# assuming a class Event exists with the following implementation
class Event:
    def __init__(self, src_id, dst_id, time, eventType):
        self.src_id = src_id
        self.dst_id = dst_id
        self.time = time
        self.eventType = eventType
        self.event_subg = {}
        
    def find_k_hop_neighbors(self, graph, node_id, k):
        start_time = time.time()
        current_nodes = {node_id}  # Start with the initial node
        all_neighbors = set(current_nodes)

        for _ in range(k):
            next_nodes = set()
            for node in current_nodes:
                # Add successors (outgoing neighbors)
                next_nodes.update(graph.successors(node).tolist())
                # Add predecessors (incoming neighbors)
                next_nodes.update(graph.predecessors(node).tolist())
            # Update the set of all neighbors
            all_neighbors.update(next_nodes)
            # Prepare for the next hop
            current_nodes = next_nodes
        end_time = time.time()
        # print("k-hop neighbors time - ", end_time-start_time)
        return all_neighbors
        
    def get_event_subg(self, graph, k=1):
        # TODO: get event affected subgraph in form of adj list
        if len(self.event_subg) != 0:
            return self.event_subg
        nodes_of_interest = torch.tensor([self.src_id, self.dst_id])
        src_neighbors = self.find_k_hop_neighbors(graph, self.src_id, k)
        dst_neighbors = self.find_k_hop_neighbors(graph, self.dst_id, k)
        all_neighbors = torch.tensor(list(src_neighbors.union(dst_neighbors)), dtype=torch.int64)
        subgraph = dgl.node_subgraph(graph, all_neighbors)
        original_ids = subgraph.ndata[dgl.NID]
        subgraph_edges = subgraph.edges()
        subgraph_edges = (
                                    original_ids[subgraph_edges[0]],  # Map source nodes
                                    original_ids[subgraph_edges[1]]  # Map destination nodes
                                  )
        src_nodes, dst_nodes = subgraph_edges
        for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
            if src not in self.event_subg:
                self.event_subg[src] = []
            self.event_subg[src].append(dst)
        # for key, value in self.event_subg.items():
        #     print(f"{key}: {value}")
        return self.event_subg

# extend the event class to include chain information as well
class DyEvent(Event):
    def __init__(self, src_id, dst_id, time, eventType, event_num, chain=None):
        
        super().__init__(src_id, dst_id, time, eventType)
        self.chain = chain
        self.event_num = event_num
    
    def get_chain(self):
        return self.chain

# dependency chain. each chain contains a list of event ids with repect to the eventList
# each chain can be executed in parallel
class DyChain:
    def __init__(self, head_event_id):
        self.eventList = []
        self.head = head_event_id
    
    def add_chain_event(self, event_id):
        self.eventList.append(event_id)
        # self.head = event_id
    
    def get_chain_head(self):
        return self.head
    
    
class DyGraph:
    def __init__(self, sampled_graph=None):
        self.dyAdjList = {}
        self.zero_event = DyEvent(-1, -1, 0, -1, None)
        self.dyAdjList[-1] = []
        # maintain a list of dyEvents
        self.eventList = []
        # maintain a list of chain-of-eventIds where each chain can be executed in parallel
        self.chainList = []
        self.sampled_graph = sampled_graph

    def add_event(self, event):
        # for key in self.dyAdjList.keys():
        #     print(key, " - ", self.dyAdjList[key])
        # print("dyadjlist keys - ", self.dyAdjList.keys())
        event_start_time = time.time()
        event_chain = None
        next_event_num = len(self.eventList)
        if(len(self.dyAdjList[-1]) == 0):
            self.dyAdjList[-1].append(next_event_num)
            # create a new chain corresponding to that event
            event_chain = DyChain(len(self.eventList))
            event_chain.add_chain_event(len(self.eventList))
            # add the new chain to the chainList
            self.chainList.append(event_chain)
        else:
            # find the head of chain to which we should add this new event
            start_time = time.time()
            head_event_id = self.find_head_event(event)
            end_time = time.time()
            # print("find_head_event time - ", end_time - start_time)
            # if no dependency with any existing events, create a new chain
            if(head_event_id == -1):
                self.dyAdjList[-1].append(next_event_num)
                event_chain = DyChain(len(self.eventList))
                event_chain.add_chain_event(len(self.eventList))
                self.chainList.append(event_chain)
            # add the new event to the chain head and update the chain
            else:
                head_event_number = self.eventList[head_event_id].event_num
                if head_event_number not in self.dyAdjList:
                    self.dyAdjList[head_event_number] = []
                # self.dyAdjList[self.eventList[head_event_id].event_num].append(event)
                self.dyAdjList[head_event_number].append(next_event_num)
                
                event_chain = self.eventList[head_event_id].get_chain()
                event_chain.add_chain_event(len(self.eventList))

        # change event to dyEvent and add it to eventList
        dyevent = DyEvent(event.src_id, event.dst_id, event.time, event.eventType, next_event_num, event_chain)
        self.eventList.append(dyevent)
        print("Event added [src, dst, number] - ", dyevent.src_id, dyevent.dst_id, dyevent.event_num)
        event_end_time = time.time()
        print("Event add time - ", event_end_time - event_start_time)

    def find_head_event(self, event):
        # for a given event, find its chain and get the head (last event) of that chain
        
        for event_id in range(len(self.eventList)):
            past_event = self.eventList[event_id]
            if(self.check_dependency(event, past_event)):
                past_event_chain = past_event.get_chain()
                head_event_id = past_event_chain.get_chain_head()
                return head_event_id
        # head event is zero event by default
        return -1
    
    def check_dependency(self, current_event, past_event):
        # check if there is a dependency b/w current event and past event
        
        event_subg = current_event.get_event_subg(self.sampled_graph)
        for node in event_subg.keys():
            # src node and dst node of past event in event triggered update set
            # if(node == past_event.src_id or node == past_event.dst_id):
            if(node == past_event.dst_id):
                return True
        return False
    
    # clear the dependency graph
    def clear(self):
        self.dyAdjList = {}
        self.zero_event = DyEvent(-1, -1, 0, None)
        self.dyAdjList[self.zero_event] = []
        self.eventList = []
        self.chainList = []
        
        
if __name__ == "__main__":
    

    # get the base and event graphs
    graphs, _ = dgl.load_graphs(args.base)
    data = graphs[0]
    graphs, _ = dgl.load_graphs(args.event)
    events_g = graphs[0]

    start_time = time.time()
    # create the dependency graph for the non-initial events
    dygraph = DyGraph(data)
    count = 0
    for edge_id in range(events_g.num_edges()):
        edge_time = events_g.edata['timestamp'][edge_id].item() 
        # if(edge_time == base_graph_time):
        #     continue
        src_node = events_g.edges()[0][edge_id].item()  # Source node
        dst_node = events_g.edges()[1][edge_id].item()  # Destination node
        
        dygraph.add_event(DyEvent(src_node, dst_node, edge_time, count, None))
        count+=1
    end_time = time.time()
    
    print("Dependency graph creation time - ", end_time - start_time)
    
    # print the events of the dependency graph
    print("Printing the Dependency Graph")
    for chain_head in dygraph.dyAdjList[-1]:
        chain_nodes = []
        chain_nodes.append(chain_head)
        if chain_head in dygraph.dyAdjList:
            for evs in dygraph.dyAdjList[chain_head]:
                chain_nodes.append(evs)
        print(chain_nodes)
            

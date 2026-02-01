"""
Network Flow Analysis
Graph-based modeling of patient flow through ED
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum
from loguru import logger

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("networkx not available, using fallback")


class EDLocation(str, Enum):
    """Standard ED locations"""
    ARRIVAL = "arrival"
    TRIAGE = "triage"
    WAITING = "waiting_room"
    BED = "ed_bed"
    LAB = "laboratory"
    IMAGING = "imaging"
    TREATMENT = "treatment"
    OBSERVATION = "observation"
    DISCHARGE = "discharge"
    ADMIT = "admission"
    TRANSFER = "transfer"
    LWBS = "lwbs"


@dataclass
class FlowEdge:
    """Edge in patient flow network"""
    source: str
    target: str
    patient_count: int = 0
    mean_duration: float = 0.0
    std_duration: float = 0.0
    probability: float = 0.0  # Transition probability
    bottleneck_score: float = 0.0
    
    
@dataclass
class FlowNode:
    """Node in patient flow network"""
    location: str
    total_visits: int = 0
    mean_time_spent: float = 0.0
    std_time_spent: float = 0.0
    current_census: int = 0
    capacity: Optional[int] = None
    utilization: float = 0.0
    is_bottleneck: bool = False
    centrality: float = 0.0


@dataclass 
class FlowMetrics:
    """Overall flow metrics"""
    total_patients: int
    mean_path_length: float
    mean_total_time: float
    bottleneck_locations: List[str]
    critical_path: List[str]
    cycle_time: float
    throughput: float
    wip: float  # Work in progress


class NetworkFlowAnalyzer:
    """
    Network Flow Analysis for ED Patient Journeys
    
    Models ED as a directed graph where:
    - Nodes = locations/stages
    - Edges = transitions between stages
    - Edge weights = transition times/probabilities
    """
    
    def __init__(self):
        self.graph: Optional[Any] = None  # networkx.DiGraph
        self.nodes: Dict[str, FlowNode] = {}
        self.edges: Dict[Tuple[str, str], FlowEdge] = {}
        self.metrics: Optional[FlowMetrics] = None
        
    def build_network_from_journeys(
        self,
        journeys: List[Dict[str, Any]]
    ) -> None:
        """
        Build flow network from patient journey data.
        
        Parameters
        ----------
        journeys : list
            List of patient journeys with events
        """
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        
        # Count transitions and durations
        transition_counts: Dict[Tuple[str, str], List[float]] = {}
        node_times: Dict[str, List[float]] = {}
        node_visits: Dict[str, int] = {}
        
        for journey in journeys:
            events = journey.get('events', [])
            if len(events) < 2:
                continue
            
            for i in range(len(events) - 1):
                source = events[i].get('location', 'unknown')
                target = events[i + 1].get('location', 'unknown')
                
                # Calculate transition time
                source_time = events[i].get('timestamp')
                target_time = events[i + 1].get('timestamp')
                
                if source_time and target_time:
                    if isinstance(source_time, str):
                        from datetime import datetime
                        source_time = datetime.fromisoformat(source_time.replace('Z', '+00:00'))
                        target_time = datetime.fromisoformat(target_time.replace('Z', '+00:00'))
                    duration = (target_time - source_time).total_seconds() / 60  # minutes
                else:
                    duration = 0.0
                
                # Record transition
                key = (source, target)
                if key not in transition_counts:
                    transition_counts[key] = []
                transition_counts[key].append(duration)
                
                # Record node visit
                node_visits[source] = node_visits.get(source, 0) + 1
                
                # Record time at source
                if source not in node_times:
                    node_times[source] = []
                node_times[source].append(duration)
            
            # Final node
            if events:
                final = events[-1].get('location', 'unknown')
                node_visits[final] = node_visits.get(final, 0) + 1
        
        # Build nodes
        all_locations = set(node_visits.keys())
        for loc in all_locations:
            times = node_times.get(loc, [0])
            self.nodes[loc] = FlowNode(
                location=loc,
                total_visits=node_visits[loc],
                mean_time_spent=np.mean(times) if times else 0,
                std_time_spent=np.std(times) if times else 0
            )
            
            if NETWORKX_AVAILABLE:
                self.graph.add_node(loc, **{
                    'visits': node_visits[loc],
                    'mean_time': np.mean(times) if times else 0
                })
        
        # Build edges
        total_from: Dict[str, int] = {}
        for (src, tgt), durations in transition_counts.items():
            total_from[src] = total_from.get(src, 0) + len(durations)
        
        for (src, tgt), durations in transition_counts.items():
            count = len(durations)
            prob = count / total_from[src] if total_from[src] > 0 else 0
            
            edge = FlowEdge(
                source=src,
                target=tgt,
                patient_count=count,
                mean_duration=np.mean(durations),
                std_duration=np.std(durations),
                probability=prob
            )
            self.edges[(src, tgt)] = edge
            
            if NETWORKX_AVAILABLE:
                self.graph.add_edge(src, tgt, weight=edge.mean_duration, **{
                    'count': count,
                    'probability': prob,
                    'mean_duration': edge.mean_duration
                })
        
        # Compute additional metrics
        self._compute_bottlenecks()
        self._compute_centrality()
        
    def _compute_bottlenecks(self) -> None:
        """Identify bottleneck locations"""
        if not self.nodes:
            return
            
        # Bottleneck score based on wait time relative to others
        mean_times = [n.mean_time_spent for n in self.nodes.values()]
        if not mean_times:
            return
            
        mean_overall = np.mean(mean_times)
        std_overall = np.std(mean_times) + 1e-6
        
        for name, node in self.nodes.items():
            z_score = (node.mean_time_spent - mean_overall) / std_overall
            node.is_bottleneck = z_score > 1.5  # More than 1.5 std above mean
            
            # Update edge bottleneck scores
            for key, edge in self.edges.items():
                if edge.source == name:
                    edge.bottleneck_score = z_score if z_score > 0 else 0
    
    def _compute_centrality(self) -> None:
        """Compute node centrality measures"""
        if not NETWORKX_AVAILABLE or not self.graph:
            return
            
        try:
            # Betweenness centrality - nodes that are on many shortest paths
            betweenness = nx.betweenness_centrality(self.graph, weight='weight')
            for name, centrality in betweenness.items():
                if name in self.nodes:
                    self.nodes[name].centrality = centrality
        except Exception as e:
            logger.warning(f"Centrality computation failed: {e}")
    
    def find_critical_path(self) -> List[str]:
        """
        Find the critical path through the ED.
        
        The critical path is the longest path from arrival to exit.
        """
        if not NETWORKX_AVAILABLE or not self.graph:
            return []
            
        # Find entry and exit nodes
        entry_nodes = ['arrival', 'triage']
        exit_nodes = ['discharge', 'admission', 'transfer', 'lwbs']
        
        entry = None
        exit_node = None
        
        for n in entry_nodes:
            if n in self.graph.nodes():
                entry = n
                break
        
        for n in exit_nodes:
            if n in self.graph.nodes():
                exit_node = n
                break
        
        if not entry or not exit_node:
            return list(self.nodes.keys())
        
        try:
            # Find longest path (critical path)
            longest = []
            longest_weight = 0
            
            for path in nx.all_simple_paths(self.graph, entry, exit_node):
                weight = sum(
                    self.graph[path[i]][path[i+1]].get('weight', 0)
                    for i in range(len(path) - 1)
                )
                if weight > longest_weight:
                    longest_weight = weight
                    longest = path
            
            return longest
            
        except nx.NetworkXNoPath:
            return []
    
    def compute_flow_metrics(self, total_hours: float = 168) -> FlowMetrics:
        """Compute overall flow metrics"""
        bottlenecks = [n.location for n in self.nodes.values() if n.is_bottleneck]
        critical_path = self.find_critical_path()
        
        # Mean path length
        if self.edges:
            total_transitions = sum(e.patient_count for e in self.edges.values())
            total_patients = sum(n.total_visits for n in self.nodes.values()) // 2  # Approx
        else:
            total_transitions = 0
            total_patients = 0
        
        mean_path_length = total_transitions / max(1, total_patients)
        
        # Mean total time
        total_time = sum(
            e.mean_duration * e.patient_count
            for e in self.edges.values()
        )
        mean_total_time = total_time / max(1, total_transitions)
        
        # Cycle time (mean time from entry to exit)
        cycle_time = sum(
            self.edges.get((critical_path[i], critical_path[i+1]), FlowEdge('', '')).mean_duration
            for i in range(len(critical_path) - 1)
        ) if len(critical_path) > 1 else 0
        
        # Throughput and WIP
        throughput = total_patients / total_hours if total_hours > 0 else 0
        wip = throughput * cycle_time / 60 if cycle_time > 0 else 0  # Little's Law
        
        self.metrics = FlowMetrics(
            total_patients=total_patients,
            mean_path_length=mean_path_length,
            mean_total_time=mean_total_time,
            bottleneck_locations=bottlenecks,
            critical_path=critical_path,
            cycle_time=cycle_time,
            throughput=throughput,
            wip=wip
        )
        
        return self.metrics
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """Get transition probability matrix"""
        locations = list(self.nodes.keys())
        n = len(locations)
        
        matrix = np.zeros((n, n))
        for i, src in enumerate(locations):
            for j, tgt in enumerate(locations):
                if (src, tgt) in self.edges:
                    matrix[i, j] = self.edges[(src, tgt)].probability
        
        return pd.DataFrame(matrix, index=locations, columns=locations)
    
    def simulate_patient_path(
        self,
        entry: str = 'arrival',
        max_steps: int = 20
    ) -> Tuple[List[str], float]:
        """
        Simulate a patient path through the network.
        
        Uses transition probabilities to generate random walk.
        """
        path = [entry]
        total_time = 0.0
        current = entry
        
        for _ in range(max_steps):
            # Get outgoing edges
            outgoing = [
                (tgt, edge.probability, edge.mean_duration)
                for (src, tgt), edge in self.edges.items()
                if src == current
            ]
            
            if not outgoing:
                break
            
            # Sample next location
            targets, probs, durations = zip(*outgoing)
            probs = np.array(probs)
            probs = probs / probs.sum()  # Normalize
            
            next_idx = np.random.choice(len(targets), p=probs)
            next_loc = targets[next_idx]
            
            path.append(next_loc)
            total_time += durations[next_idx]
            current = next_loc
            
            # Check for exit
            if current in ['discharge', 'admission', 'transfer', 'lwbs']:
                break
        
        return path, total_time
    
    def get_network_data(self) -> Dict[str, Any]:
        """Get network data for visualization"""
        return {
            "nodes": [
                {
                    "id": name,
                    "label": name.replace('_', ' ').title(),
                    "visits": node.total_visits,
                    "mean_time": round(node.mean_time_spent, 1),
                    "is_bottleneck": node.is_bottleneck,
                    "centrality": round(node.centrality, 3)
                }
                for name, node in self.nodes.items()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "count": edge.patient_count,
                    "mean_duration": round(edge.mean_duration, 1),
                    "probability": round(edge.probability, 3)
                }
                for edge in self.edges.values()
            ],
            "metrics": {
                "total_patients": self.metrics.total_patients if self.metrics else 0,
                "mean_path_length": round(self.metrics.mean_path_length, 2) if self.metrics else 0,
                "cycle_time": round(self.metrics.cycle_time, 1) if self.metrics else 0,
                "throughput": round(self.metrics.throughput, 2) if self.metrics else 0,
                "bottlenecks": self.metrics.bottleneck_locations if self.metrics else [],
                "critical_path": self.metrics.critical_path if self.metrics else []
            }
        }

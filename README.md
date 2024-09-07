

## Overview

This repository contains the implementation and evaluation of various network optimization algorithms developed as part of my bachelor thesis. The algorithms were extensively tested to assess their performance in realistic network scenarios. The tests were conducted using different tools and datasets to ensure reliable results.

## Testing Environments

### 1. SNDLib / TopologyZoo ( in BA-1 -File)

For simulations, real network topologies from SNDLib and TopologyZoo were used. These sources provide a variety of network architectures, allowing the algorithms to be tested in realistic scenarios. Each algorithm was evaluated at the graph level, with Python and the NetworkX and NetworKit libraries used for calculating network metrics and shortest paths.

- **Traffic Generation:** Traffic was generated using the MCF method. Here, 20% of the connections were chosen randomly, and the demand was adjusted to ensure that the link utilization was always at 100%. This allowed for realistic testing of the algorithms.
- **Datasets:** Each algorithm was tested with 10 different datasets to obtain reliable results.

### 2. Mininet  o ( in Mininet Test -File)

Mininet is an open-source tool for network simulation. It allows the creation of virtual networks with hosts, switches, and links on a single machine. Various network topologies can be created to test the algorithms.

- **Algorithm Test:** We test the "Dynamic Routing with Failover 4" algorithm in Mininet to assess connection stability in a virtual network. Different scenarios, such as connection failures, are simulated. Four nodes are used in the network, and continuous data is sent from node A to node D. An additional node is added to maintain the connection and ensure data flow in case of a node failure.

### 3. Nanonet o ( in Nanonet Test -File)

Nanonet is a virtualized network environment concept based on Mininet. It allows for network simulation through the use of network namespaces in the Linux kernel and the creation of (virtual) connections between these namespaces.

- **Algorithm Test:** The idealwaypointOp algorithm is tested in a scenario with four demands from nodes 11, 12, 13, and 14. Each node sends a flow that starts simultaneously and lasts for 300 seconds, aiming to reach node 4. The network consists of nine nodes plus an additional node (node 5). The goal of the test is to determine how well the algorithm utilizes network resources and how effectively the links are utilized. The results will show how the algorithm performs and can be optimized under realistic conditions.


## Tools Used

- **Python:** For implementing and analyzing the algorithms.
- **NetworkX and NetworKit:** For calculating network metrics and shortest paths.
- **Mininet:** For simulating virtual networks.
- **Nanonet:** For further simulation and optimization of network resources.


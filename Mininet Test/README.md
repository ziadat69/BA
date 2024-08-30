# Mininet Network Test Script

This repository contains a Python script that simulates a network topology using Mininet. The script sets up a network with five switches and five hosts and tests the stability and routing capabilities of the network under various link conditions.

## Overview

The script performs the following tasks:

1. **Network Topology Creation**: Creates a network topology with five switches (`s1` to `s5`) and five hosts (`A`, `B`, `C`, `D`, and `G`).
2. **Link Utilization Monitoring**: Measures the link utilization between specific hosts using `iperf`.
3. **Link Failure Simulation**: Disables specific links to simulate failures and tests the network's ability to recover by reconfiguring routing paths.
4. **Ping Test**: Checks connectivity between hosts using `ping` and displays the network path if the ping is successful.
5. **Link Status Checking**: Monitors the status of the links and updates routing as needed.

## Prerequisites

- **Mininet**: Ensure Mininet is installed on your system. Installation instructions can be found [here](http://mininet.org/download/).
- **iperf**: The script uses `iperf` to measure link utilization. Install `iperf` on all Mininet hosts.
- **Open vSwitch (OVS)**: You need to install Open vSwitch for network switching.

### Installing Open vSwitch

To install Open vSwitch on Ubuntu, follow these steps:

1. **Update your package list**:
   
    ```bash
    sudo apt-get update
    ```

2. **Install Open vSwitch**:
   
    ```bash
    sudo apt-get install openvswitch-switch
    ```

3. **Verify the Installation**:
   
    ```bash
    sudo systemctl status openvswitch-switch
    ```

   Ensure the service is running. You should see output indicating that the Open vSwitch service is active.

## How to Run the Test

### Running the Script

1. **Clone the Repository**:
   
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2. **Execute the Mininet Script**:
   
    ```bash
    sudo python3 mininet_test.py
    ```

3. **Observe the Results**:
   - The script sets up the network topology and simulates link failures.
   - It performs automatic pings and routing tests. Check the console output to see the results.

4. **Manual Testing with CLI**:
   - After the script completes, a Mininet CLI will open. You can perform additional tests manually by running commands such as `ping`, `traceroute`, and `iperf`.

5. **Troubleshooting**:
   - If the test fails or does not work as expected, check the console output for error messages.
   - You can also manually activate or deactivate links using the CLI to test specific scenarios.

### Developing the Script Further

1. **Topology Adjustments**:
   - Add additional hosts or switches by extending the `topology()` function.
   - Example: Add a new host `H` and link it to one of the existing switches:
   
   ```python
   H = net.addHost('H', ip='10.0.0.6/24')
   net.addLink(H, s2)

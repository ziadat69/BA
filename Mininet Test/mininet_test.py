from mininet.net import Mininet
from mininet.node import OVSController, OVSSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from time import sleep

def test_ping_and_print_route(src, dest):
    """
    Test connectivity between src and dest using ping, and print the route if successful.
    """
    result = src.cmd('ping -c 10 %s' % dest.IP())
    if '0% packet loss' in result:
        print(result)
        print(f"Ping from {src.name} to {dest.name} succeeded.")
        route_result = src.cmd('traceroute %s' % dest.IP())
        print(f"Route from {src.name} to {dest.name}:\n{route_result}")
        return True
    else:
        print(f"Ping from {src.name} to {dest.name} failed.")
        return False

def is_link_down(h1, h2):
    """
    Check if the link between h1 and h2 is down using ping.
    """
    result = h1.cmd('ping -c 1 %s' % h2.IP())
    if '1 packets transmitted, 1 received' in result:
        return False  # Link is up
    else:
        return True  # Link is down

def calculate_link_utilization(h1, h2):
    """
    Measure link utilization between h1 and h2 using iperf.
    """
    h2.cmd('iperf -s -u -i 1 > /dev/null &')
    result = h1.cmd('iperf -c %s -u -b 1M -t 1' % h2.IP())
    h2.cmd('pkill iperf')
    utilization = extract_utilization(result)
    return utilization

def extract_utilization(result):
    """
    Extract utilization value from iperf results.
    """
    for line in result.splitlines():
        if "Mbits/sec" in line:
            return float(line.split()[-2])
    return 0

def check_all_links_and_update_routing(net):
    """
    Check specified link pairs and update routing if necessary.
    """
    link_down = False
    A, B, C, D = net.get('A', 'B', 'C', 'D')
    link_pairs = [
        (A, B),
        (B, C),
        (D, C)
    ]
    
    for h1, h2 in link_pairs:
        utilization = calculate_link_utilization(h1, h2)
        print(f"Link between {h1.name} and {h2.name} is using {utilization} Mbps.")
        
        if is_link_down(h1, h2):
            print(f"The link between {h1.name} and {h2.name} is down!")
            link_down = True

    if link_down:
       return True
    

def topology():
    """
    Create and test the network topology.
    """
    net = Mininet(controller=OVSController, switch=OVSSwitch, link=TCLink)

    # Add the controller
    net.addController('c0')

    # Add switches
    s1 = net.addSwitch('s1')
    s2 = net.addSwitch('s2')
    s3 = net.addSwitch('s3')
    s4 = net.addSwitch('s4')
    s5 = net.addSwitch('s5')

    # Add hosts
    A = net.addHost('A', ip='10.0.0.1/24')
    B = net.addHost('B', ip='10.0.0.2/24')
    C = net.addHost('C', ip='10.0.0.3/24')
    D = net.addHost('D', ip='10.0.0.4/24')
    G = net.addHost('G', ip='10.0.0.5/24')

    # Add links
    net.addLink(A, s1)
    net.addLink(s1, s2)
    net.addLink(s2, B)
    net.addLink(s2, s3)
    net.addLink(s3, C)
    net.addLink(s3, s4)
    net.addLink(s4, D)
    net.addLink(s5, G)
    net.addLink(s1, s5)
    net.addLink(s2, s5)
    net.addLink(s3, s5)
    net.addLink(s4, s5)

    # Start the network
    net.start()

    # Disable some links initially
    net.configLinkStatus('s1', 's5', 'down')
    net.configLinkStatus('s2', 's5', 'down')
    net.configLinkStatus('s3', 's5', 'down')
    net.configLinkStatus('s4', 's5', 'down')

    # Disable a link for testing
    print("Disabling the link between s2 and s3 for testing...")
    net.configLinkStatus('s2', 's3', 'down')
    
    net.pingAll()
    test_ping_and_print_route(A, D) 
    if test_ping_and_print_route(A, D):
       net.configLinkStatus('s1', 's5', 'up')
       net.configLinkStatus('s4', 's5', 'up') 
    try:
        while True:
            check_all_links_and_update_routing(net)
            
            if test_ping_and_print_route(A, D):
                net.pingAll()
                print("Ping successful, exiting...")
                net.stop()
                return
            sleep(5)
    except KeyboardInterrupt:
        print("Stopping the checks.")
    # Run CLI for manual testing
    CLI(net)
    net.stop()

if __name__ == '__main__':
    topology()



# README

## Übersicht

Dieses Repository enthält die Implementierung und Evaluierung verschiedener Netzwerkoptimierungsalgorithmen, die im Rahmen meiner Bachelorarbeit entwickelt wurden. Die Algorithmen wurden umfassend getestet, um ihre Leistung in realistischen Netzwerkszenarien zu bewerten. Die Tests wurden mit verschiedenen Tools und Datensätzen durchgeführt, um zuverlässige Ergebnisse zu erzielen.

## Testumgebungen

### 1. SNDLib / TopologyZoo ( in BA-1 -File)
 
Für die Simulationen wurden reale Netzwerktopologien aus SNDLib und TopologyZoo verwendet. Diese Quellen bieten eine Vielzahl von Netzwerkarchitekturen, die es ermöglichen, die Algorithmen in realistischen Szenarien zu testen. Jeder Algorithmus wurde auf Graphenebene evaluiert, wobei Python und die Bibliotheken NetworkX und NetworKit zur Berechnung von Netzwerkmetriken und kürzesten Pfaden verwendet wurden.

- **Verkehrserzeugung:** Der Verkehr wurde mit der MCF-Methode erstellt. Hierbei wurden zufällig 20% der Verbindungen gewählt und die Nachfrage so angepasst, dass die Auslastung der Links immer bei 100% liegt. Dies ermöglichte eine realistische Testung der Algorithmen.
- **Datensätze:** Jeder Algorithmus wurde mit 10 verschiedenen Datensätzen getestet, um zuverlässige Ergebnisse zu erhalten.

### 2. Mininet o ( in Mininet Test -File)

Mininet ist ein Open-Source-Tool zur Simulation von Netzwerken. Es ermöglicht das Erstellen von virtuellen Netzwerken mit Hosts, Switches und Links auf einer einzigen Maschine. Verschiedene Netzwerktopologien können erstellt werden, um die Algorithmen zu testen.

- **Algorithmus-Test:** Wir überprüfen den Algorithmus „Dynamisches Routing mit Ausfallschutz 4“ in Mininet, um die Stabilität der Verbindung in einem virtuellen Netzwerk zu testen. Verschiedene Szenarien, wie der Ausfall von Verbindungen, werden simuliert. Dazu verwenden wir vier Knoten im Netzwerk und senden kontinuierlich Daten von Knoten A zu Knoten D. Ein zusätzlicher Knoten wird hinzugefügt, um die Verbindung aufrechtzuerhalten und den Datenfluss bei einem Knoten-Ausfall sicherzustellen.

### 3. Nanonet o ( in Nanonet Test -File)

Nanonet ist ein virtualisiertes Netzwerkumgebungskonzept, das auf Mininet basiert. Es ermöglicht die Simulation von Netzwerken durch die Nutzung von Netzwerk-Namensräumen (Namespaces) im Linux-Kernel und die Schaffung von virtuellen Verbindungen zwischen diesen Namensräumen.

- **Algorithmus-Test:** Der idealwaypointOp-Algorithmus wird in einem Szenario mit vier Anforderungen von den Knoten 11, 12, 13 und 14 getestet. Jeder Knoten sendet einen Flow, der gleichzeitig startet und 300 Sekunden dauert, um Knoten 4 zu erreichen. Das Netzwerk besteht aus neun Knoten plus einem zusätzlichen Knoten (Knoten 5). Ziel des Tests ist es, herauszufinden, wie gut der Algorithmus die Netzwerkressourcen nutzt und wie effektiv die Links ausgelastet werden. Die Ergebnisse sollen zeigen, wie der Algorithmus unter realistischen Bedingungen funktioniert und optimiert werden kann.




## Verwendete Tools

- **Python:** Zur Implementierung und Analyse der Algorithmen.
- **NetworkX und NetworKit:** Zur Berechnung von Netzwerkmetriken und kürzesten Pfaden.
- **Mininet:** Zur Simulation von virtuellen Netzwerken.
- **Nanonet:** Zur weiteren Simulation und Optimierung von Netzwerkressourcen.


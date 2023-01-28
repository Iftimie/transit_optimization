import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import numpy as np
import heapq
import torch
from collections import defaultdict
import random
from copy import deepcopy

def get_data():

    np.random.seed(0)

    NUM_DATAPOINTS_PER_CLUSTER = 20
    # Define cluster centers for sources
    c1 = [[3, 1], [1, 3]]
    c2 = [[6, -5], [-5, 6]]
    c3 = [[6, 10], [-10, 6]]

    # Define cluster covariances (spreading) for sources
    ssc = [
        ((10, 10),c1),
        ((10, 20),c2),
        ((20, 30),c3),
        ((1, 5),c1)
    ]

    # Define cluster centers for destinations
    c1 = [[10, 1], [1, 10]]
    c2 = [[6, -5], [-5, 6]]
    c3 = [[16, 10], [-10, 16]]

    # Define cluster covariances (spreading) for destinations
    ddc = [
        ((10, 1),c3),
        ((20, 10),c1),
        ((1, 30),c2),
        ((1, 5),c3)
    ]

    sources = []
    destinations = []
    for (s, sc), (d, dc) in zip(ssc, ddc):
        source = np.random.multivariate_normal(s, sc, NUM_DATAPOINTS_PER_CLUSTER)
        sources.append(source)

        dest = np.random.multivariate_normal(d, dc, NUM_DATAPOINTS_PER_CLUSTER)
        destinations.append(dest)

    # normalize coordinates to be between 0 and 1
    sources = np.concatenate(sources, axis=0)
    destinations = np.concatenate(destinations, axis=0)
    all_ = np.concatenate((sources, destinations), axis=0)
    minx, maxx = all_[:, 0].min(), all_[:, 0].max()
    miny, maxy = all_[:, 1].min(), all_[:, 1].max()
    sources[:, 0] = (sources[:, 0] - minx) / (maxx-minx)
    sources[:, 1] = (sources[:, 1] - miny) / (maxy-miny)
    destinations[:, 0] = (destinations[:, 0] - minx) / (maxx-minx)
    destinations[:, 1] = (destinations[:, 1] - miny) / (maxy-miny)

    np.random.shuffle(destinations)

    return sources, destinations

def get_dummy_busstations(sources, destinations):
    all_ = np.concatenate((sources, destinations), axis=0)
    minx, maxx = all_[:, 0].min(), all_[:, 0].max()
    miny, maxy = all_[:, 1].min(), all_[:, 1].max()
    NUM_LINES_X = 3
    NUM_LINES_Y = 4
    xs = np.linspace(minx, maxx, num=NUM_LINES_X)
    DX = xs[1] - xs[0]
    ys = np.linspace(miny, maxy, num=NUM_LINES_Y)
    DY = ys[1] - ys[0] if len(ys) >= 2 else DX
    # lines = list(range(NUM_LINES_Y + NUM_LINES_X))
    cmap = get_cmap(NUM_LINES_X+NUM_LINES_Y)
    line_finding_x = {
        x: i for i, x in enumerate(xs)
    }
    line_finding_y = {
        y: NUM_LINES_X+i for i, y in enumerate(ys)
    }


    nodes = []
    import itertools
    for i, n in enumerate(itertools.product(xs,ys)):
        nodes.append((i, n))

    nodearray = np.array(list(n for i, n in nodes))

    return nodearray, nodes, line_finding_x, line_finding_y, DX, DY

def connect_bus_stations(nodes, line_finding_x, line_finding_y):
    BUS_SPEED = 10

    from collections import defaultdict
    g = defaultdict(list)

    for i, node in nodes:
        x, y = node

        # right
        rightx = float('inf')
        sj = None
        for j, nnode in nodes:
            nx, ny = nnode
            if i!=j and ny == y and nx > x and nx < rightx:
                sj = j
                rightx = nx
        
        if sj:
            _, (nx, ny) = nodes[sj]
            d = (((x - nx)**2 + (y - ny)**2)**0.5) / BUS_SPEED
            line = line_finding_y[y]
            g[i].append((sj, d, line))

        # left
        leftx = float('-inf')
        sj = None
        for j, nnode in nodes:
            nx, ny = nnode
            if i!=j and ny == y and nx < x and nx > leftx:
                sj = j
                leftx = nx
        if sj:
            _, (nx, ny) = nodes[sj]
            line = line_finding_y[y]
            d = (((x - nx)**2 + (y - ny)**2)**0.5) / BUS_SPEED
            g[i].append((sj, d, line))
        # up
        upy = float('inf')
        sj = None
        for j, nnode in nodes:
            nx, ny = nnode
            if i!=j and nx == x and ny > y and ny < upy:
                sj = j
                upy = ny
        if sj:
            _, (nx, ny) = nodes[sj]
            line = line_finding_x[x]
            d = (((x - nx)**2 + (y - ny)**2)**0.5) / BUS_SPEED
            g[i].append((sj, d, line))
        # down
        downy = float('-inf')
        sj = None
        for j, nnode in nodes:
            nx, ny = nnode
            if i!=j and nx == x and ny < y and ny > downy:
                sj = j
                downy = ny
        if sj:
            _, (nx, ny) = nodes[sj]
            d = (((x - nx)**2 + (y - ny)**2)**0.5) / BUS_SPEED
            line = line_finding_x[x]
            g[i].append((sj, d, line))

    return g

def add_source_and_destination_to_graph(g, sources, destinations, nodes, DX, DY, line_finding_x, line_finding_y):
    # add to the graph
    sources_offset = len(nodes)
    for i, source in enumerate(sources):
        nodes.append((sources_offset + i, source))

    dest_offset = len(nodes)
    for i, dest in enumerate(destinations):
        nodes.append((dest_offset + i, dest))


    for i, source in enumerate(sources):
        source_x, source_y = source
        dest_x, dest_y = destinations[i]
        # connect every source directly to the destination
        # if NUM_LINES_Y > 1:
        #     d = ((source_x - dest_x)**2 + (source_y - dest_y)**2)**0.5
        #     g[sources_offset + i].append((dest_offset + i, d))
        for node_index, node_coord in nodes[:sources_offset]:
            # connect every source directly to every bus station only if in proximity
            bus_station_x, bus_station_y = node_coord
            if (abs(source_x - bus_station_x) < DX and abs(source_y - bus_station_y) < DY):
                d = ((source_x - bus_station_x)**2 + (source_y - bus_station_y)**2)**0.5
                g[sources_offset + i].append((node_index, d, line_finding_x[bus_station_x]))
                g[sources_offset + i].append((node_index, d, line_finding_y[bus_station_y]))

    for node_index, node_coord in nodes[:sources_offset]:
        # connect every bus station directly to a destination only if in proximity
        for i, dest in enumerate(destinations):
            bus_station_x, bus_station_y = node_coord
            dest_x, dest_y = dest
            if abs(dest_x - bus_station_x) < DX and abs(dest_y - bus_station_y) < DY:
                d = ((dest_x - bus_station_x)**2 + (dest_y - bus_station_y)**2)**0.5
                g[node_index].append((dest_offset + i, d, -1))
                g[node_index].append((dest_offset + i, d, -1))
    return sources_offset, dest_offset, g

def get_cmap(n, name='nipy_spectral'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

BUS_SPEED = 50
class TorchGraph:
    def __init__(self, nodes, sources_offset, dest_offset, g, sources, destinations, cmap) -> None:
        self.sources = sources
        self.destinations = destinations
        self.cmap = cmap
        self.nodesgt = []
        self.sources_offset = sources_offset
        self.dest_offset = dest_offset
        for i, node in nodes:
            requires_grad = True
            if i >= sources_offset:
                requires_grad = False

            self.nodesgt.append(torch.tensor(node, requires_grad=requires_grad))

        g = deepcopy(g)
        for k in g:
            g[k] = list((item[0], item[2]) for item in g[k])
        
        self.g = g

    def calc(self, s, d):
        distance = torch.linalg.norm(self.nodesgt[s] - self.nodesgt[d])
        if s < self.sources_offset and d < self.sources_offset:
            distance = distance / BUS_SPEED
        return distance


    def dijkstragt(self, src, dst):
        heap = []
        visited = set()
        heapq.heappush(heap, (0, src, -1, [(src, -1)], 0))
        ans = torch.tensor(100000.0)
        track = []
        while heap: 
            cost, v, line, p, distance = heapq.heappop(heap)
            visited.add(v)

            if v == dst and cost < ans:
                ans = cost
                track = p
                continue

            for to in self.g[v]:
                if to[0] in visited:
                    continue
                distance = self.calc(v, to[0])
                newcost = cost+distance
                if line != to[1]:
                    newcost += 0.05 # the cost of waiting the bus. either waiting or changing or descencing
                heapq.heappush(heap, (newcost, to[0], to[1], p + [to], distance))

        return ans, track



    def animate_track(self):
        busstations = np.array(list(n.detach().numpy() for n in self.nodesgt if n.requires_grad))
        
        for i in random.sample(range(len(self.sources)),3):
            # f.clear()
            # plt.ion()
            f = plt.figure()
            plt.scatter(self.sources[i, 0], self.sources[i, 1], color='orange')
            plt.scatter(self.destinations[i, 0], self.destinations[i, 1], color='blue')
            plt.scatter(busstations[:, 0], busstations[:, 1], color='green')
            res, track = self.dijkstragt(self.sources_offset + i, self.dest_offset + i)
            cnode = self.dest_offset + i
            for currentnode, nextnode in zip(track, track[1:]):
                cnode, cline = currentnode
                nnode, nline = nextnode
                if cline == -1 or nline == -1:
                    c = 'black'
                else:
                    c = self.cmap(nline)
                
                destnode = self.nodesgt[cnode]
                sourcenode = self.nodesgt[nnode]
                x, y = sourcenode.detach().numpy()
                nx, ny = destnode.detach().numpy()

                plt.plot([x, nx],[y, ny],  linewidth=0.5, c=c)
                # plt.pause(0.3)
            # plt.pause(1)

    def get_line_stats(self):
        lines_stats = defaultdict(int)
        
        for i in range(len(self.sources)):
            res, track = self.dijkstragt(self.sources_offset + i, self.dest_offset + i)
            cnode = self.dest_offset + i
            for currentnode, nextnode in zip(track[1:], track[2:-1]): # excluding source and destinati
                cnode, cline = currentnode
                nnode, nline = nextnode
                lines_stats[(cnode, nnode, nline)] += 1
                lines_stats[(nnode, cnode, nline)] += 1

        return lines_stats
    
    def optimize(self):
        params = [n for n in self.nodesgt if n.requires_grad == True]
        optimizer = torch.optim.SGD(params, lr=0.005)

        fig = plt.figure()

        for i in range(3):

            optimizer.zero_grad()
            total = torch.tensor(0.0)
            for i in range(len(self.sources)):
                res, track = self.dijkstragt(self.sources_offset + i, self.dest_offset + i)
                total += res
            print(total)
            total.backward()
            optimizer.step()

            new_bus_stations = np.array([p.detach().numpy() for p in params])

            fig.clear()
            plt.ion()
            plt.scatter(new_bus_stations[:, 0], new_bus_stations[:, 1], color='green')
            for i in range(self.sources_offset):
                x, y = new_bus_stations[i]
                for j, line in self.g[i]:
                    if j > self.sources_offset:
                        continue
                    node = new_bus_stations[j]
                    nx, ny = node
                    plt.plot([x, nx],[y, ny],  linewidth=0.4, c=self.cmap(line))

            plt.pause(0.1)
            # plt.close('all')
            # plt.show(block=False)
        plt.show()

    def plot_bus_line_usage(self):
        line_stats = self.get_line_stats()

        bus_lines_usage = defaultdict(int)
        for i in range(len(self.sources)):
            res, track = self.dijkstragt(self.sources_offset + i, self.dest_offset + i)
            for t in track:
                bus_lines_usage[t[1]]+=1

        def map_from_on_internal_to_another(value, leftMin, leftMax, rightMin, rightMax):
            leftSpan = leftMax - leftMin
            rightSpan = rightMax - rightMin

            valueScaled = float(value - leftMin) / float(leftSpan)

            return rightMin + (valueScaled * rightSpan)

        max_count, min_count = max(line_stats.values()), min(line_stats.values())

        plt.figure()
        plt.ioff()
        for k, v in line_stats.items():
            cnode, nnode, line = k
            if line == -1:
                continue
            c = self.cmap(line)

            destnode = self.nodesgt[cnode]
            sourcenode = self.nodesgt[nnode]
            x, y = sourcenode.detach().numpy()
            nx, ny = destnode.detach().numpy()

            width = map_from_on_internal_to_another(v, min_count, max_count, 1, 4.0)

            plt.text((x+nx)/2, (y+ny) / 2, str(v), fontsize=8)

            plt.plot([x, nx],[y, ny],  linewidth=width, c=c)
        plt.show()
import os
import re
import argparse
import pickle
import networkx as nx
import urllib.request as request
import pandas as pd
import numpy as np

def readEntryList(filename, sep='[\s,; ]+'):
    """
    read a text file contains a list of objects ( ligand names / pdb ids / pubchem ids ), 
    which are seperated by common delimiter ( \tab, \space, comma, etc. )
    """
    with open(filename, 'r') as file:
        content = file.read()
    results = re.split(sep, content)
    results = [r.strip() for r in results]
    return results


def download_single_file(url, dest, overwrite=True):
    if os.path.exists(dest) and not overwrite:
        print(f"File: {dest} already exists.")
    else:
        try:
            request.urlretrieve(url, dest)
        except Exception as e:
            print(f'Fail to download {url} to {dest}.')
            print(e)


def fetch_clusters(sqid, outdir, overwrite):
    assert type(sqid)==int and sqid in [30, 40, 50, 70, 90, 95, 100]
    url = f'https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{sqid}.txt'
    dest = os.path.join(outdir, f'clusters-by-entity-{sqid}.txt')
    download_single_file(url, dest, overwrite)


def mapfunc(item):
    i, j, cluster_dct = item
    x = cluster_dct[i]
    y = cluster_dct[j]
    if len(x)!=0 and len(y)!=0:
        it = set(x).intersection(set(y))
        u = set(x).union(set(y))
        sim = len(it) / len(u)
        if sim >= np.finfo(float).eps:
            return (i, j, sim)
    return None


def make_graph_clans(folder, sqid, pdbids, overwrite=True):
    clsfile = os.path.join(folder, f'clusters-by-entity-{sqid}.txt')
    nxfile = os.path.join(folder, f'ClanGraph_{sqid}.pkl')
    if os.path.exists(nxfile) and not overwrite:
        print(f"File: {nxfile} already exists.")
        with open(nxfile, 'rb') as file:
            clanG = pickle.load(file)
    else:
        if not os.path.exists(clsfile):
            raise FileNotFoundError(f'{clsfile} does not exist.')
        with open(clsfile, 'r') as file:
            cluster_dct = {}
            for i, line in enumerate(file.readlines()):
                pids_in_cluster = [j.split("_")[0].lower() for j in line.strip().split()]
                if pdbids == 'all':
                    pids_in_cluster = sorted(set(pids_in_cluster))
                elif type(pdbids) == list:
                    pids_in_cluster = sorted(set(pids_in_cluster).intersection(set(pdbids)))
                else:
                    raise TypeError("""TypeError: 'pdbids' should either be 
                        'all' or a list of pdbids.
                        function: make_graph_clans(folder, sqid, pdbids, overwrite=True)""")
                if len(pids_in_cluster) == 0:
                    continue
                else:
                    cluster_dct[i] = pids_in_cluster
        # calculate cluster edges
        clutser_nodes = list(cluster_dct.keys())
        cluster_jaccard = list(map(mapfunc, [(i, j, cluster_dct) for i in clutser_nodes for j in clutser_nodes if i < j]))
        cluster_jaccard = [item for item in cluster_jaccard if item is not None]
        clanG = nx.Graph()
        # add cluster-pid edges
        for i, pids in cluster_dct.items():
            clanG.add_weighted_edges_from([(i, pid, 1) for pid in pids])
        # add cluster-cluster edges
        clanG.add_weighted_edges_from(cluster_jaccard)
        with open(nxfile, 'wb') as file:
            pickle.dump(clanG, file)
    return clanG


def make_cluster_clans(clanG, dffile, overwrite):
    if os.path.exists(dffile) and not overwrite:
        print(f"File: {dffile} already exists.")
        return
    clans = list(nx.connected_components(clanG))
    clans = sorted([[node for node in clan if type(node)==str] for clan in clans], key=len, reverse=True)
    clans = [sorted(i) for i in clans]
    
    
    df = pd.DataFrame([(idx, clan) for idx, clan in enumerate(clans)], \
        columns=['Structure_Clan_ID', 'PDBIDList'])
    with open(dffile, 'wb') as file:
        pickle.dump(df, file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Cluster protein structures based on sequence homology.
    """)
    parser.add_argument('--sqid', type=int, default=90, \
        help='sequence cutoff')
    parser.add_argument('--outdir', type=str, \
        help='folder to store downloaded sequence clutering result and calculation outputs')
    parser.add_argument('--pdbids', type=str, default='all', \
        help='select all pdbids in cluster file, or provide a file containing pdb ids interested')
    parser.add_argument('--overwrite', action='store_true', default=False, \
        help='Overwrite existing files if flagged')
    parser.add_argument('--draw', action='store_true', default=False, \
        help='Whether to draw the cluster graph')
    args = parser.parse_args()

    fetch_clusters(args.sqid, args.outdir, args.overwrite)
    if args.pdbids == 'all':
        pdbids = 'all'
    else:
        pdbids = readEntryList(args.pdbids)
        pdbids = [pid.lower() for pid in pdbids]
    
    clanG = make_graph_clans(args.outdir, args.sqid, pdbids, overwrite=args.overwrite)
    # `Structure_Clan_ID`, `PDBID list`
    dffile = os.path.join(args.outdir, f'ClanGraph_{args.sqid}_df.pkl')
    make_cluster_clans(clanG, dffile, args.overwrite)
    
    if args.draw:
        import matplotlib.pyplot as plt
        pngfile = os.path.join(args.outdir, f'ClanGraph_{args.sqid}.png')
        fig, ax = plt.subplots(figsize=(12, 12))
        # Visualize graph components
        # pos = nx.spring_layout(clanG, weight='weight')
        pos = nx.nx_agraph.graphviz_layout(clanG, prog="neato")
        edgecolor = []
        edgelength = []
        for u, v in clanG.edges():
            if type(u)==int and type(v)==int:
                # two cluster nodes
                edgecolor.append('lightgray')
            else:
                # cluster - pdbid edge
                edgecolor.append('black')
        nodecolor = []
        nodesize = []
        for node in clanG.nodes():
            if type(node)==int:
                nodecolor.append('blue')
                nodesize.append(10)
            else: 
                nodecolor.append('green')
                nodesize.append(2)
        
        nx.draw_networkx_edges(clanG, pos, alpha=0.3, edge_color=edgecolor)
        nx.draw_networkx_nodes(clanG, pos, node_color=nodecolor, node_size=nodesize, alpha=0.9)
        ax.margins(0.1, 0.05)
        fig.tight_layout()
        plt.axis("off")
        plt.savefig(pngfile, dpi=600)


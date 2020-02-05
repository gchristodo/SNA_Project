#from community import modularity
import networkx as nx
import matplotlib.pyplot as plt
#from community import modularity
from datetime import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import warnings
import warnings as _warnings
import networkx as nx
#from networkx.utils.decorators import not_implemented_for
#from ...utils import arbitrary_element





    #@uthor Sidiras
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import warnings

# only with 2 generation manual.... initial popoulation and the first generation ---> afterr in a looop

warnings.simplefilter(action='ignore', category=FutureWarning)



# for notebook remove below  #
# %pylab inline
# @uthor sidirasg


    # neiborlist`
def nebigorList(G, no):
    allne = list(G.neighbors(no))
    return allne


    # mutation replace a comunity random with other

def mutation(chromosum, nodes):
    nl = list(nodes)
    random.shuffle(nl)  # take randomly one node
    rplc = nl.pop()
    indxrp = random.randint(0, len(chromosum))
    chromosum[indxrp] = rplc  # replace the random node with the  random aother node
    return chromosum


# Take the neibor list @DataFrame for each nodeq
def neb(z):
    nodeneibdf = pd.DataFrame()
    a = []
    n = []
    for i in z:
        g = nebigorList(z, i)
        a.append(g)
        n.append(i)
    nodeneibdf['node'] = n
    nodeneibdf['neighbors'] = a
    return nodeneibdf

popoul=60

    # now we have to find the maximuc cc for each nebor
    # and
def ccNodeVal(g, cc):
    nodeL = []
    km = []
    for c in cc['neighbors']:
        k = 0
        ko = 0
        node1 = ''
        tempko = []
    for i in c:

        k = (nx.clustering(g, i))
        # print('node', c, ' has neibors ', i,' and the cc is ',k)
        if (k > ko):
            ko = k
            node1 = i
        elif (k == ko):
            tempko.append(i)
    # random choose a neibor if the have the maximu cc then choose random a node
    if len(tempko) > 0:
        node1 = random.choice(tempko)
# these prints are only for debuging to notify the values

# print(" The maximou CC is node",node1, 'with maximum cc ',ko,tempko)

    nodeL.append(node1)
    km.append(ko)

    # Appent To Dataset the CC node and cc Value
# locus-based adjacency representation we have node and mmCCNode and is 'supergen' with high cc value mod1
    cc['mmCCNode'] = nodeL
    cc['mmCCval'] = km


clusercoeff = pd.DataFrame()


def ccnode(G):
    clusercoeff = pd.DataFrame()
    nodl = []
    ccl = []

    for ni in nodes:
        nodl.append(ni)
        ccl.append((nx.clustering(G, ni)))
    clusercoeff['node'] = nodl
    clusercoeff['cc'] = ccl
    return clusercoeff


def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst) - 1, 2)}
    return res_dct


# import deep
# This is the cross over function pick a random parant
def Ucross(par1, par2):
    out = []
    for i in range(0, len(par1)):  # TODOneed try cach in case the length of par1!=par2
        condi = bool(random.getrandbits(1))  # condi random Out True or false
        if condi == True:
            out.append(par1[i])
        else:
            out.append(par2[i])
    return out


# now we have to find the maximuc cc for each nebor
# and
def ccNodeVal(g, cc):
    nodeL = []
    km = []
    for c in cc['neighbors']:
        k = 0
        ko = 0
        node1 = ''
        tempko = []
        for i in c:

            k = (nx.clustering(g, i))
            # print('node', c, ' has neibors ', i,' and the cc is ',k)

            if (k > ko):
                ko = k
                node1 = i
            elif (k == ko):
                tempko.append(i)
        # random choose a neibor if the have the maximu cc then choose random a node
        if len(tempko) > 0:
            node1 = random.choice(tempko)
        # these prints are only for debuging to notify the values

        # print(" The maximou CC is node",node1, 'with maximum cc ',ko,tempko)

        nodeL.append(node1)
        km.append(ko)

    # Appent To Dataset the CC node and cc Value
    # locus-based adjacency representation we have node and mmCCNode and is 'supergen' with high cc value mod1
    cc['mmCCNode'] = nodeL
    cc['mmCCval'] = km


def ccnode(G):
    clusercoeff = pd.DataFrame()
    nodl = []
    ccl = []

    for ni in nodes:
        nodl.append(ni)
        ccl.append((nx.clustering(G, ni)))
    clusercoeff['node'] = nodl
    clusercoeff['cc'] = ccl
    return clusercoeff


def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst) - 1, 2)}
    return res_dct


##########
# we will use this function to produce random papulation, only if they are  are neibors
def populationn(num, datfr):
    global chromosom
    while num > 0:
        for i in datfr:
            k = list(i)
            random.shuffle(k)
            chromosom.append(k.pop())
        tempna = 'chromosom' + str(num)
        popoulationInit[tempna] = chromosom
        chromosom = []
        # name = 'G' + str(tempna)
        num = num - 1

G = nx.karate_club_graph()
pos = nx.spring_layout(G)
# True labels of the group each student (node) unded up in. Found via the original paper
 # we have 2 clusters one with node 0 a
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]




    #take alal the neiboras of the graph to
cc=neb(G)

clusercoeff = pd.DataFrame()

    #run and append @Dataset for each node the max node CC with the value CC
ccNodeVal(G,cc)



    # all nodes
nodesA = list(nx.nodes(G))



def CrossFirstGen():
    for i in range(1, popoulationInit.shape[1] - 1):
        ran = random.randint(1, (popoulationInit.shape[1] - 2))  # is the random  number
        chrRan = 'chromosom' + str(ran)
        nameg = 'offspring' + str(i)
        chr = 'chromosom' + str(i)
        tmp2 = (Ucross(list(popoulationInit[chr]), popoulationInit[chrRan]))
        offspiring['chromosom' + str(i)] = tmp2
        tmp2 = []


def evalgen1():
    # ----------------------------------------------------------------------------result= pd.DataFrame()
    gp = []
    for i in range(0, offspiring.shape[1] - 2):  # we have 2 extra coloum 0 coloumn and node coloumn
        chro = 'chromosom' + str(i)
        gp.append(nx.from_pandas_edgelist(offspiring, chro, 'node'))
    for i in gp:
        # gr=gp[i]
        concomponet.append(modularity(i, list(list(nx.connected_components(i)))))
    result['modularity'] = concomponet
    result['grpah'] = gp
    # take finala result the result for next generation
    resultF = result[result['modularity'] > 0.65]
    indx = chromosom + resultF.index.values.tolist()
    bbf['node'] = cc['node']
    return result
"""Functions for measuring the quality of a partition (into
communities).
"""

from functools import wraps
from itertools import product

import networkx as nx
from networkx import NetworkXError
from networkx.utils import not_implemented_for
from networkx.algorithms.community.community_utils import is_partition

__all__ = ['coverage', 'modularity', 'performance']


class NotAPartition(NetworkXError):
    """Raised if a given collection is not a partition.
    """
    def __init__(self, G, collection):
        msg = f"{G} is not a valid partition of the graph {collection}"
        super().__init__(msg)


def require_partition(func):
    """Decorator to check that a valid partition is input to a function
    Raises :exc:`networkx.NetworkXError` if the partition is not valid.
    This decorator should be used on functions whose first two arguments
    are a graph and a partition of the nodes of that graph (in that
    order)::
        >>> @require_partition
        ... def foo(G, partition):
        ...     print('partition is valid!')
        ...
        >>> G = nx.complete_graph(5)
        >>> partition = [{0, 1}, {2, 3}, {4}]
        >>> foo(G, partition)
        partition is valid!
        >>> partition = [{0}, {2, 3}, {4}]
        >>> foo(G, partition)
        Traceback (most recent call last):
          ...
        networkx.exception.NetworkXError: `partition` is not a valid partition of the nodes of G
        >>> partition = [{0, 1}, {1, 2, 3}, {4}]
        >>> foo(G, partition)
        Traceback (most recent call last):
          ...
        networkx.exception.NetworkXError: `partition` is not a valid partition of the nodes of G
    """
    @wraps(func)
    def new_func(*args, **kw):
        # Here we assume that the first two arguments are (G, partition).
        if not is_partition(*args[:2]):
            raise nx.NetworkXError('`partition` is not a valid partition of'
                                   ' the nodes of G')
        return func(*args, **kw)
    return new_func


def intra_community_edges(G, partition):
    """Returns the number of intra-community edges for a partition of `G`.
    Parameters
    ----------
    G : NetworkX graph.
    partition : iterable of sets of nodes
        This must be a partition of the nodes of `G`.
    The "intra-community edges" are those edges joining a pair of nodes
    in the same block of the partition.
    """
    return sum(G.subgraph(block).size() for block in partition)


def inter_community_edges(G, partition):
    """Returns the number of inter-community edges for a prtition of `G`.
    according to the given
    partition of the nodes of `G`.
    Parameters
    ----------
    G : NetworkX graph.
    partition : iterable of sets of nodes
        This must be a partition of the nodes of `G`.
    The *inter-community edges* are those edges joining a pair of nodes
    in different blocks of the partition.
    Implementation note: this function creates an intermediate graph
    that may require the same amount of memory as that of `G`.
    """
    # Alternate implementation that does not require constructing a new
    # graph object (but does require constructing an affiliation
    # dictionary):
    #
    #     aff = dict(chain.from_iterable(((v, block) for v in block)
    #                                    for block in partition))
    #     return sum(1 for u, v in G.edges() if aff[u] != aff[v])
    #
    MG = nx.MultiDiGraph if G.is_directed() else nx.MultiGraph
    return nx.quotient_graph(G, partition, create_using=MG).size()


def inter_community_non_edges(G, partition):
    """Returns the number of inter-community non-edges according to the
    given partition of the nodes of `G`.
    `G` must be a NetworkX graph.
    `partition` must be a partition of the nodes of `G`.
    A *non-edge* is a pair of nodes (undirected if `G` is undirected)
    that are not adjacent in `G`. The *inter-community non-edges* are
    those non-edges on a pair of nodes in different blocks of the
    partition.
    Implementation note: this function creates two intermediate graphs,
    which may require up to twice the amount of memory as required to
    store `G`.
    """
    # Alternate implementation that does not require constructing two
    # new graph objects (but does require constructing an affiliation
    # dictionary):
    #
    #     aff = dict(chain.from_iterable(((v, block) for v in block)
    #                                    for block in partition))
    #     return sum(1 for u, v in nx.non_edges(G) if aff[u] != aff[v])
    #
    return inter_community_edges(nx.complement(G), partition)


@not_implemented_for('multigraph')
@require_partition
def performance(G, partition):
    """Returns the performance of a partition.
    The *performance* of a partition is the ratio of the number of
    intra-community edges plus inter-community non-edges with the total
    number of potential edges.
    Parameters
    ----------
    G : NetworkX graph
        A simple graph (directed or undirected).
    partition : sequence
        Partition of the nodes of `G`, represented as a sequence of
        sets of nodes. Each block of the partition represents a
        community.
    Returns
    -------
    float
        The performance of the partition, as defined above.
    Raises
    ------
    NetworkXError
        If `partition` is not a valid partition of the nodes of `G`.
    References
    ----------
    .. [1] Santo Fortunato.
           "Community Detection in Graphs".
           *Physical Reports*, Volume 486, Issue 3--5 pp. 75--174
           <https://arxiv.org/abs/0906.0612>
    """
    # Compute the number of intra-community edges and inter-community
    # edges.
    intra_edges = intra_community_edges(G, partition)
    inter_edges = inter_community_non_edges(G, partition)
    # Compute the number of edges in the complete graph (directed or
    # undirected, as it depends on `G`) on `n` nodes.
    #
    # (If `G` is an undirected graph, we divide by two since we have
    # double-counted each potential edge. We use integer division since
    # `total_pairs` is guaranteed to be even.)
    n = len(G)
    total_pairs = n * (n - 1)
    if not G.is_directed():
        total_pairs //= 2
    return (intra_edges + inter_edges) / total_pairs


@require_partition
def coverage(G, partition):
    """Returns the coverage of a partition.
    The *coverage* of a partition is the ratio of the number of
    intra-community edges to the total number of edges in the graph.
    Parameters
    ----------
    G : NetworkX graph
    partition : sequence
        Partition of the nodes of `G`, represented as a sequence of
        sets of nodes. Each block of the partition represents a
        community.
    Returns
    -------
    float
        The coverage of the partition, as defined above.
    Raises
    ------
    NetworkXError
        If `partition` is not a valid partition of the nodes of `G`.
    Notes
    -----
    If `G` is a multigraph, the multiplicity of edges is counted.
    References
    ----------
    .. [1] Santo Fortunato.
           "Community Detection in Graphs".
           *Physical Reports*, Volume 486, Issue 3--5 pp. 75--174
           <https://arxiv.org/abs/0906.0612>
    """
    intra_edges = intra_community_edges(G, partition)
    total_edges = G.number_of_edges()
    return intra_edges / total_edges


def modularity(G, communities, weight='weight'):
    r"""Returns the modularity of the given partition of the graph.
    Modularity is defined in [1]_ as
    .. math::
        Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_ik_j}{2m}\right)
            \delta(c_i,c_j)
    where $m$ is the number of edges, $A$ is the adjacency matrix of
    `G`, $k_i$ is the degree of $i$ and $\delta(c_i, c_j)$
    is 1 if $i$ and $j$ are in the same community and 0 otherwise.
    Parameters
    ----------
    G : NetworkX Graph
    communities : list or iterable of set of nodes
        These node sets must represent a partition of G's nodes.
    Returns
    -------
    Q : float
        The modularity of the paritition.
    Raises
    ------
    NotAPartition
        If `communities` is not a partition of the nodes of `G`.
    Examples
    --------
    >>> import networkx.algorithms.community as nx_comm
    >>> G = nx.barbell_graph(3, 0)
    >>> nx_comm.modularity(G, [{0, 1, 2}, {3, 4, 5}])
    0.35714285714285704
    >>> nx_comm.modularity(G, nx_comm.label_propagation_communities(G))
    0.35714285714285704
    References
    ----------
    .. [1] M. E. J. Newman *Networks: An Introduction*, page 224.
       Oxford University Press, 2011.
    """
    if not isinstance(communities, list):
        communities = list(communities)
    if not is_partition(G, communities):
        raise NotAPartition(G, communities)

    multigraph = G.is_multigraph()
    directed = G.is_directed()
    m = G.size(weight=weight)
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        norm = 1 / m
    else:
        out_degree = dict(G.degree(weight=weight))
        in_degree = out_degree
        norm = 1 / (2 * m)

    def val(u, v):
        try:
            if multigraph:
                w = sum(d.get(weight, 1) for k, d in G[u][v].items())
            else:
                w = G[u][v].get(weight, 1)
        except KeyError:
            w = 0
        # Double count self-loops if the graph is undirected.
        if u == v and not directed:
            w *= 2
        return w - in_degree[u] * out_degree[v] * norm

    Q = sum(val(u, v) for c in communities for u, v in product(c, repeat=2))
    return Q * norm





# graph node and mmCCnode
# graph node and mmCCnode
G2 = nx.from_pandas_edgelist(cc, 'node', 'mmCCNode')
#local brebjes
#nx.draw_networkx(G2)
#plt.show()

# find the conencted componets of the new graph and is the chromosome
# the initial popoulation with2  is concomp
concomp = list(list(nx.connected_components(G2)))

numCluster = len(concomp)


def CrossFirstGen():
    for i in range (1,popoulationInit.shape[1]-1):
        ran=random.randint(1, (popoulationInit.shape[1] - 2))#is the random  number
        chrRan='chromosom'+str(ran)
        nameg = 'offspring' + str(i)
        chr = 'chromosom' + str(i)
        tmp2 = (Ucross(list(popoulationInit[chr]), popoulationInit[chrRan]))
        offspiring['chromosom' + str(i)] = tmp2
        tmp2 = []


# @uthor sidirasg


# neiborlist`
def nebigorList(G, no):
    allne = list(G.neighbors(no))
    return allne


# mutation replace a comunity random with other

def mutation(chromosum, nodes):
    nl = list(nodes)
    random.shuffle(nl)  # take randomly one node
    rplc = nl.pop()
    indxrp = random.randint(0, len(chromosum))
    chromosum[indxrp] = rplc  # replace the random node with the  random aother node
    return chromosum


# Take the neibor list @DataFrame for each nodeq
def neb(z):
    nodeneibdf = pd.DataFrame()
    a = []
    n = []
    for i in z:
        g = nebigorList(z, i)
        a.append(g)
        n.append(i)
    nodeneibdf['node'] = n
    nodeneibdf['neighbors'] = a
    return nodeneibdf


# now we have to find the maximuc cc for each nebor
# and
def ccNodeVal(g, cc):
    nodeL = []
    km = []
    for c in cc['neighbors']:
        k = 0
        ko = 0
        node1 = ''
        tempko = []
        for i in c:

            k = (nx.clustering(g, i))
            # print('node', c, ' has neibors ', i,' and the cc is ',k)

            if (k > ko):
                ko = k
                node1 = i
            elif (k == ko):
                tempko.append(i)
        # random choose a neibor if the have the maximu cc then choose random a node
        if len(tempko) > 0:
            node1 = random.choice(tempko)
        # these prints are only for debuging to notify the values

        # print(" The maximou CC is node",node1, 'with maximum cc ',ko,tempko)

        nodeL.append(node1)
        km.append(ko)

    # Appent To Dataset the CC node and cc Value
    # locus-based adjacency representation we have node and mmCCNode and is 'supergen' with high cc value mod1
    cc['mmCCNode'] = nodeL
    cc['mmCCval'] = km

# Take the neibor list @DataFrame for each nodeq
def neb(z):
    nodeneibdf = pd.DataFrame()
    a = []
    n = []
    for i in z:
        g = nebigorList(z, i)
        a.append(g)
        n.append(i)
    nodeneibdf['node'] = n
    nodeneibdf['neighbors'] = a
    return nodeneibdf


#nx.draw_networkx(G)
#plt.show()
mutationRate = 0.04
prob = 0.5  # prob of croos over
nodesA = list(nx.nodes(G))


#take alal the neiboras of the graph to
cc=neb(G)

clusercoeff = pd.DataFrame()

#run and append @Dataset for each node the max node CC with the value CC
ccNodeVal(G,cc)



# all nodes
nodesA = list(nx.nodes(G))



########################################################################################################################
# take alal the neiboras of the graph to
cc = neb(G)

# run and append @Dataset for each node the max node CC with the value CC
ccNodeVal(G, cc)

# Show @Dat aframe cc we have the neibors list and te cc

# graph node and mmCCnode
# graph node and mmCCnode

#local brebjes
#nx.draw_networkx(G2)
#plt.show()

# find the conencted componets of the new graph and is the chromosome
# the initial popoulation with2  is concomp
concomp = list(list(nx.connected_components(G2)))
#concomp = list(list(nx.connected_components(G2)))

numCluster = len(concomp)







#evalouation each chromosome
#we have to bult a graph for each cromosome and find the conected comunites
#then we run modularity and storees it

def evalouation_Gen1():
    result=pd.DataFrame()
    concomponet=[]
    def evalgen1():

    #----------------------------------------------------------------------------result= pd.DataFrame()
        gp=[]
        for i in range (0,offspiring.shape[1]-2): #we have 2 extra coloum 0 coloumn and node coloumn
            chro = 'chromosom' + str(i)
            gp.append(nx.from_pandas_edgelist(offspiring, chro, 'node'))
        for i in gp:
        #gr=gp[i]
            concomponet.append(modularity(i,list(list(nx.connected_components(i)))))
        result['modularity']=concomponet
        result['grpah']=gp
    #take finala result the result for next generation
        resultF=result[result['modularity']>0.65]
        indx=resultF.index.values.tolist()
        bbf['node']=cc['node']
        return result
    #evalouation category
    res1=evalgen1()
    res1A=result[result['modularity']>0.70]

    TotalLen=len(res1A)+len(res1B)
    #0.1 times popoulation the good chromosmes of the popoulation is 30 then we take 3 times each best chromosomys
    times=int(popoul*0.1)
    ins=[]
    for i in range(0,times):
        for k in res1A.index.values.tolist():
            offspiringN['chromosom' + str(100*(k+i))]=offspiring['chromosom' + str(k)]
            ins.append('chromosom' + str(100*(k+i)))#indexing

    result=result.sort_values(by=['modularity'],ascending=False)
    creted=offspiringN.shape[1]-1
    TotalNextPop=60-creted
    #Take the first modularity chromosems is about /
#selection we take the first chromosemes with hig modulartity 3 or more times and with low modularity only one time
    resNext=result.sort_values(by=['modularity'],ascending=False).head(TotalNextPop)
    #now we  create or select the popoulation with low modularity
    indx=[]
    for k in resNext.index.values.tolist():
        offspiringN['chromosom' + str((k))]=offspiring['chromosom' + str(k)]
        indx.append('chromosom'+str(k))
    return indx+ins,resNext
#/////////////////////////////////////////////////////////////////////////////////////////////Rep
# /////start
import random

# modularity for all chromosume
mod1 = modularity(G2, concomp)
# create fist initail popoulation where there are neibors the
popoulationInit = pd.DataFrame()
chromosom = []
# we create 1st generation manual manualy with best chrom
offspiring = pd.DataFrame()
offspiringN = pd.DataFrame()
offspiring['chromosom0'] = cc['mmCCNode']
offspiring['node'] = cc['node']
# offspiringN['chromosom0']=cc['mmCCNode']
offspiringN['node'] = cc['node']
result = pd.DataFrame()
resultF = pd.DataFrame()
bbf = pd.DataFrame()
res1 = pd.DataFrame()
res1A = pd.DataFrame()
TotalLen = pd.DataFrame()
res1B = pd.DataFrame()
result = pd.DataFrame()

import pandas as pd

offspiring['chromosom0'] = cc['mmCCNode']
offspiring['node'] = cc['node']
offspiringN['chromosom0'] = cc['mmCCNode']
offspiringN['node'] = cc['node']  # cluster coefficeint cromosmo
populationn(popoul, cc['neighbors'])

# fist crosever at offssping gataset
CrossFirstGen()

evalouatfirtst = evalouation_Gen1()

indL = evalouatfirtst[0]

res1 = evalouatfirtst[1]

import random

for generation in range(1, 100):

    print("Generation:", generation)
    # cross over second generation in a loop

    # evalouatfirtst[0]
    # indL=evalouatfirtst[0]
    # indx=evalouatfirtst[0]
    prob = 0.6  # probability to  cross over beteen two chromosomata

    offspiring = pd.DataFrame()  # first Dataset
    inx3 = []  # we have different structurs when the inex is oiut of the loop from inital population
    # offspiring['node']=cc['node']
    prob = 0.6  # probability of cross over
    # global indL
    offspiring = pd.DataFrame()  # second dataset
    inx3 = []

    for i in indL:
        inx3.append(str(i))
        ran = random.choice(indL)  # is the random  number
        # we have different structurs when the inex is oiut of the loop from inital population
        # offspiring['node']=cc['node']
        #
    chrRan = random.choice(indL)
    # nameg = 'offspring' + str(i)
    # chr = 'chromosom' + str(i)
    rando = random.randint(1, 100)  # random 1 to 100
    perh = prob * 100  # if is > from 100 then cross
    if perh > rando:
        rando = random.randint(1, 100)
        tmp2 = (Ucross(list(offspiringN[i]), list(offspiringN[ran])))
        offspiring[str(i)] = offspiringN[str(i)]
    else:  # if we don have cross over the chromosom tdirect to the new generation
        offspiring[str(i)] = offspiringN[str(i)]

    offspiring['node'] = cc['node']
    result = pd.DataFrame()
    concomponet = []
    gp = []
    indx=[]

    for i in inx3:  # we have 2 extra coloum 0 coloumn and node coloumn
        chro = inx3
        gp.append(nx.from_pandas_edgelist(offspiringN, 'node', i))
        #concomponet.append(modularity(gp[i], list(nx.connected_components(gp[q]))))
        indx.append(i)



    for q in gp:
        # gr=gp[i]
        concomponet.append(modularity(q, list(nx.connected_components(q))))



    result['modularity'] = concomponet
    result['grpah'] = gp
    result['indx'] = i
    # take finala result the result for next generation
    resultF = result[result['modularity'] > 0.65]
    indx = chromosom + resultF.index.values.tolist()
    # bbf['node']=cc['node']
    indx = chromosom + resultF.index.values.tolist()
    res1 = result['modularity']
    # resulat modularity res1=evalgen1()
    res1A = result[result['modularity'] > 0.70]
    res1B = result[result['modularity'] < 0.70]
    TotalLen = len(res1A) + len(res1B)
    # 0.1 times popoulation the good chromosmes of the popoulation is 30 then we take 3 times each best chromosomys
    times = int(popoul * 0.1)
    ins = []
    for i in range(0, times):
        for k in res1A.indx.values.tolist():
            offspiringN[str(k)] = offspiring[str(k)]
            ins.append(str(k))  # indexing
            result = result.sort_values(by=['modularity'], ascending=False)
    creted = offspiringN.shape[1] - 1
    TotalNextPop = 60 - creted
    # Take the first modularity chromosems is about /
    # selection we take the first chromosemes with hig modulartity 3 or more times and with low modularity only one time
    resNext = result.sort_values(by=['modularity'], ascending=False).head(TotalNextPop)  # the best vhjromosum
    # now we  create or select the popoulation with low modularity
    indx = []
    for k in resNext.indx.values.tolist():
        offspiringN[str((k))] = offspiring[str(k)]
        indx.append(str(k))
        # return indx+ins,resNext

        #######
        # mutation
        # mutation
        # fint he curent popoulation
    curpup = int(offspiringN.shape[1] * mutationRate)

    nummutation = int(mutationRate * offspiringN.shape[1])
    rinxx = indx[:]  # copy list
    muta = []
    if len(indx)>0:
        while curpup>=0:
            random.shuffle(rinxx) #take random %persent times ffrom ofspring N mutatioon
            try:
                muta.append(rinxx.pop())
            except:
                print("Mutation perfomed")
            curpup=curpup-1
        for i in muta:   #implemet the mutation
            offspiringN[i] = mutation(offspiringN[i], nodesA)  #replece a random gen  that is goint to cross to the next generation
            curpup=curpup-1







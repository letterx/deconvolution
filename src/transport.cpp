#include "transport.hpp"

#include <assert.h>
#include <limits>
#include <vector>

#include <boost/intrusive/list.hpp>
#include <boost/intrusive/slist.hpp>

template <typename T>
class TransportGraph {
    public:
        TransportGraph(int sizeSupply, int sizeDemand)
            : _sizeSupply(sizeSupply)
            , _sizeDemand(sizeDemand)
            , _nodes(sizeSupply+sizeDemand)
            , _treeEdges(2*(sizeSupply+sizeDemand - 1))
        { 
            for (int i = 0; i < _sizeSupply+_sizeDemand; ++i)
                _nodes[i].id = i;
        }
        
        typedef boost::intrusive::list_base_hook<
            boost::intrusive::link_mode<boost::intrusive::normal_link>
            > ListHook;
        struct Edge : public ListHook { 
            int target;
            int edgeIdx;
            Edge* reverse;
        };

        typedef boost::intrusive::list<Edge, 
                boost::intrusive::base_hook<ListHook>> EdgeList;

        typedef boost::intrusive::slist_member_hook<
            boost::intrusive::link_mode<boost::intrusive::normal_link>
            > StackHook;

        struct Node {
            int id = 0;
            T potential = 0;
            EdgeList outEdges;
            typename EdgeList::iterator dfsOutEdge;
            StackHook stackHook;
            bool inDfsStack = false;
        };
        typedef boost::intrusive::slist<Node,
                boost::intrusive::member_hook<Node, 
                                              StackHook, 
                                              &Node::stackHook>
                > NodeStack;


        void addTreeEdge(int supply, int demand) {
            assert(_numTreeEdges+2 <= static_cast<int>(_treeEdges.size()));
            Edge& e1 = _treeEdges[_numTreeEdges];
            Edge& e2 = _treeEdges[_numTreeEdges+1];
            _numTreeEdges += 2;

            int edgeIdx = supply*_sizeDemand+demand;
            e1.target = demand+_sizeSupply;
            e1.edgeIdx = edgeIdx;
            e1.reverse = &e2;
            e2.target = supply;
            e2.edgeIdx = edgeIdx;
            e2.reverse = &e1;

            _nodes[supply].outEdges.push_back(e1);
            _nodes[demand+_sizeSupply].outEdges.push_back(e2);
        }

        void augment(int supply, int demand, T* flows) {
            int edgeIdx = supply*_sizeDemand+demand;

            for (auto& n : _nodes)
                n.inDfsStack = false;

            NodeStack stack;
            {
                Node& startNode = _nodes.at(supply);
                stack.push_front(startNode);
                startNode.dfsOutEdge = startNode.outEdges.begin();
                startNode.inDfsStack = true;
            }
            while (!stack.empty() && stack.front().id != demand+_sizeSupply) {
                auto& n = stack.front();
                if (n.dfsOutEdge == n.outEdges.end()) {
                    stack.pop_front();
                    assert(!stack.empty());
                    auto& nextN = stack.front();
                    assert(nextN.dfsOutEdge != nextN.outEdges.end());
                    assert(nextN.dfsOutEdge->target == n.id);
                    nextN.dfsOutEdge++;
                    continue;
                } else {
                    int target = n.dfsOutEdge->target;
                    auto& nextN = _nodes.at(target);
                    if (nextN.inDfsStack) {
                        n.dfsOutEdge++;
                        continue;
                    } else {
                        stack.push_front(nextN);
                        nextN.dfsOutEdge = nextN.outEdges.begin();
                        nextN.inDfsStack = true;
                        continue;
                    }
                }
            }
            assert(!stack.empty());
            assert(stack.front().id == demand+_sizeSupply);
            assert((stack.size() % 2) == 0);
            stack.pop_front();

            T bottleneckFlow = std::numeric_limits<T>::max();
            Edge* bottleneckEdge = nullptr;
            bool parity = true;
            for (auto& n : stack) {
                if (parity) {
                    auto& e = *n.dfsOutEdge;
                    auto f = flows[e.edgeIdx];
                    if (f < bottleneckFlow) {
                        bottleneckFlow = f;
                        bottleneckEdge = &e;
                    }
                }
                parity = !parity;
            }
            assert(bottleneckFlow < std::numeric_limits<T>::max());
            assert(bottleneckEdge != nullptr);

            parity = true;
            for (auto& n : stack) {
                auto& e = *n.dfsOutEdge;
                if (parity)
                    flows[e.edgeIdx] -= bottleneckFlow;
                else
                    flows[e.edgeIdx] += bottleneckFlow;
                parity = !parity;
            }
            flows[edgeIdx] += bottleneckFlow;

            Edge& e1 = *bottleneckEdge;
            Edge& e2 = *e1.reverse;
            {
                Node& n1 = _nodes[e2.target];
                Node& n2 = _nodes[e1.target];
                n1.outEdges.erase(n1.outEdges.iterator_to(e1));
                n2.outEdges.erase(n2.outEdges.iterator_to(e2));
            }
            {
                Node& n1 = _nodes[supply];
                Node& n2 = _nodes[demand+_sizeSupply];
                e1.target = demand+_sizeSupply;
                e1.edgeIdx = edgeIdx;
                e1.reverse = &e2;
                e2.target = supply;
                e2.edgeIdx = edgeIdx;
                e2.reverse = &e1;
                n1.outEdges.push_back(e1);
                n2.outEdges.push_back(e2);
            }
        }

        void updatePotentials(const T* costs) {
            for (auto& n : _nodes) {
                n.potential = 0;
                n.inDfsStack = false;
            }

            NodeStack stack;
            {
                Node& startNode = _nodes.at(0);
                stack.push_front(startNode);
                startNode.dfsOutEdge = startNode.outEdges.begin();
                startNode.inDfsStack = true;
            }
            while (!stack.empty()) {
                auto& n = stack.front();
                if (n.dfsOutEdge == n.outEdges.end()) {
                    stack.pop_front();
                    if(!stack.empty()) {
                        auto& nextN = stack.front();
                        assert(nextN.dfsOutEdge != nextN.outEdges.end());
                        assert(nextN.dfsOutEdge->target == n.id);
                        nextN.dfsOutEdge++;
                    }
                    continue;
                } else {
                    int target = n.dfsOutEdge->target;
                    auto& nextN = _nodes.at(target);
                    if (nextN.inDfsStack) {
                        n.dfsOutEdge++;
                        continue;
                    } else {
                        int edgeIdx = n.dfsOutEdge->edgeIdx;
                        T c = costs[edgeIdx];
                        nextN.potential = c - n.potential;

                        stack.push_front(nextN);
                        nextN.dfsOutEdge = nextN.outEdges.begin();
                        nextN.inDfsStack = true;
                        continue;
                    }
                }
            }
        }

        std::tuple<int, int, T> findPivot(const T* costs) {
            std::tuple<int, int, T> minEdge 
                = {0, 0, std::numeric_limits<T>::max()};
            for (int i = 0; i < _sizeSupply; ++i) {
                for (int j = 0; j < _sizeDemand; ++j) {
                    int edgeIdx = i*_sizeDemand+j;
                    T resCost = costs[edgeIdx] 
                        - _nodes[i].potential 
                        - _nodes[j+_sizeSupply].potential;
                    if (resCost < std::get<2>(minEdge))
                        minEdge = {i, j, resCost};
                }
            }
            return minEdge;
        }


        int _sizeSupply;
        int _sizeDemand;
        int _numTreeEdges = 0;
        std::vector<Node> _nodes;
        std::vector<Edge> _treeEdges;
};

template <typename T>
void solveTransport(int sizeSupply, int sizeDemand, const T* costs, 
        const T* supply, const T* demand, T* flow) {
    T sumSupply = 0;
    for (int i = 0; i < sizeSupply; ++i) 
        sumSupply += supply[i];
    T sumDemand = 0;
    for (int j = 0; j < sizeDemand; ++j)
        sumDemand += demand[j];
    assert(sumSupply == sumDemand);

    for (int k = 0; k < sizeSupply*sizeDemand; ++k)
        flow[k] = 0;

    std::vector<T> resSupply(supply, supply+sizeSupply);
    std::vector<T> resDemand(demand, demand+sizeDemand);

    TransportGraph<T> graph{sizeSupply, sizeDemand};

    int numNodes = sizeSupply+sizeDemand;
    for (int k = 0; k < numNodes-1; ++k) {
        T minCost = std::numeric_limits<T>::max();
        int minI = 0;
        int minJ = 0;
        for (int i = 0; i < sizeSupply; ++i) {
            if (resSupply.at(i) == 0)
                continue;
            for (int j = 0; j < sizeDemand; ++j) {
                if (resDemand.at(j) == 0)
                    continue;
                if (costs[i*sizeDemand+j] < minCost) {
                    minCost = costs[i*sizeDemand+j];
                    minI = i;
                    minJ = j;
                }
            }
        }
        T f = std::min(resSupply.at(minI), resDemand.at(minJ));
        resSupply.at(minI) -= f;
        resDemand.at(minJ) -= f;
        flow[minI*sizeDemand+minJ] += f;
        graph.addTreeEdge(minI, minJ);
    }

    for (auto s : resSupply)
        assert(s == 0);
    for (auto d : resDemand)
        assert(d == 0);

    while (true) {
        graph.updatePotentials(costs);
        std::tuple<int, int, T> minEdge = graph.findPivot(costs);
        if (std::get<2>(minEdge) < 0)
            graph.augment(std::get<0>(minEdge), std::get<1>(minEdge), flow);
        else
            break;
    }
}


#define INSTANTIATE_TRANSPORT(T) \
    template void solveTransport<T>(int sizeSupply, int sizeDemand, \
            const T* costs, const T* supply, const T* demand, T* flow);
INSTANTIATE_TRANSPORT(double)
INSTANTIATE_TRANSPORT(int32_t)
INSTANTIATE_TRANSPORT(int64_t)
#undef INSTANTIATE_TRANSPORT

#ifndef _MST_HXX_
#define _MST_HXX_

#include <utility>
#include <boost/shared_array.hpp>

struct UnionFind {

    struct Entry {
        int parent;
        int depth;
        int size;
    };

    int N;
    boost::shared_array<Entry> partitions;

    UnionFind(int N = 0);

    int Find(int elem) const;
    int Depth(int elem) const { return partitions[Find(elem)].depth; }
    int ComponentSize(int elem) const { return partitions[Find(elem)].size; }
    void Merge(int e1, int e2);
    int Size() const { return N; }
};

inline UnionFind::UnionFind(int N)
    : N(N) 
{
    partitions = boost::shared_array<Entry>(new Entry[N]);
    for (int i = 0; i < N; ++i) {
        partitions[i].parent = i;
        partitions[i].depth = 1;
        partitions[i].size = 1;
    }
}

inline int UnionFind::Find(int elem) const {
    int parent = partitions[elem].parent;
    while (parent != elem) {
        elem = parent;
        parent = partitions[elem].parent;
    }
    return elem;
}

inline void UnionFind::Merge(int e1, int e2) {
    int p1 = Find(e1);
    int p2 = Find(e2);
    if (p1 == p2)
        return;
    int d1 = partitions[p1].depth;
    int d2 = partitions[p2].depth;
    if (d1 < d2) {
        partitions[p1].parent = p2;
        partitions[p2].size += partitions[p1].size;
    } else if (d2 < d1) {
        partitions[p2].parent = p1;
        partitions[p1].size += partitions[p2].size;
    } else {
        partitions[p2].parent = p1;
        partitions[p1].size += partitions[p2].size;
        partitions[p1].depth++;
    }
}

#endif

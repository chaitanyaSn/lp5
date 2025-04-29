#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

// Graph using adjacency list
class Graph
{
public:
    int V;
    vector<vector<int>> adj;

    Graph(int V)
    {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v)
    {
        adj[u].push_back(v);
        adj[v].push_back(u); // undirected graph
    }
};

// Parallel BFS
void parallelBFS(Graph &g, int start)
{
    vector<bool> visited(g.V, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty())
    {
        int size = q.size();
        vector<int> current_level;

        for (int i = 0; i < size; ++i)
        {
            int node = q.front();
            q.pop();
            cout << node << " ";
            current_level.push_back(node);
        }

        vector<int> next_level;

        #pragma omp parallel for
        for (int i = 0; i < current_level.size(); ++i)
        {
            int u = current_level[i];
            for (int v : g.adj[u])
            {
                bool mark = false;
                #pragma omp critical
                {
                    if (!visited[v])
                    {
                        visited[v] = true;
                        mark = true;
                    }
                }
                if (mark)
                {
                    #pragma omp critical
                    next_level.push_back(v);
                }
            }
        }

        for (int v : next_level)
        {
            q.push(v);
        }
    }
    cout << endl;
}

// Parallel DFS using OpenMP tasks
void parallelDFSUtil(Graph &g, int u, vector<bool> &visited)
{
    #pragma omp critical
    {
        if (visited[u]) return;
        visited[u] = true;
        cout << u << " ";
    }

    for (int v : g.adj[u])
    {
        bool shouldVisit = false;

        #pragma omp critical
        {
            if (!visited[v])
                shouldVisit = true;
        }

        if (shouldVisit)
        {
            // Properly scoped task
            #pragma omp task shared(g, visited)
            {
                parallelDFSUtil(g, v, visited);
            }
        }
    }

    #pragma omp taskwait
}

void parallelDFS(Graph &g, int start)
{
    vector<bool> visited(g.V, false);
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallelDFSUtil(g, start, visited);
        }
    }
    cout << endl;
}

// Main function
int main()
{
    Graph g(7);

    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);

    cout << "Parallel BFS starting from node 0:" << endl;
    parallelBFS(g, 0);

    cout << "Parallel DFS starting from node 0:" << endl;
    parallelDFS(g, 0);

    return 0;
}


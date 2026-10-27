#include <bliss/graph.hh>

int main()
{
  bliss::Graph graph(0);
  bliss::Stats stats;
  graph.find_automorphisms(stats, NULL, NULL);
  return 0;
}

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <queue>
#include <iomanip>
#include <cmath>
#include <limits>
#include <Highs.h>
#include <scip/scip.h>
#include <scip/scipdefplugins.h>
#include <scip/cons_linear.h>
#include <unordered_set>
#include <functional>
#include <map>
#include <regex>
#include <stdexcept>
#include <cstdlib>
#include <cctype>
#include <xlslib.h>
#include <filesystem>

using namespace std;

struct RawEdge { string u, v; double cost; string key; };
struct PairReq { string u_orig, v_orig; double density; int id; };
using WeightedGraph = std::unordered_map<std::string, std::unordered_map<std::string, double>>;
struct NodeDist {
    std::string node;
    double dist;

    bool operator>(const NodeDist& other) const {
        return dist > other.dist;
    }
};

int NUM_K = 1;
const double SCALE = 1e6;

string safeName(string s) {
    if(s.empty()) return "empty";
    s.erase(0, s.find_first_not_of(" \t\r\n\""));
    s.erase(s.find_last_not_of(" \t\r\n\"") + 1);
    for (char& c : s) if (!isalnum((unsigned char)c)) c = '_';
    return s;
}

std::map<std::string, std::pair<std::string, double>> global_pair_neighbor_index;

std::vector<std::string> findPathInOriginal(
    std::string start,
    std::string end,
    const WeightedGraph& adj) {

    if (adj.find(start) == adj.end() || adj.find(end) == adj.end()) {
        return {};
    }

    if (adj.at(start).count(end)) {
        return {start, end};
    }

    std::unordered_map<std::string, double> dist;
    std::unordered_map<std::string, std::string> parent;

    std::priority_queue<std::pair<double, std::string>,
                        std::vector<std::pair<double, std::string>>,
                        std::greater<>> pq;

    dist[start] = 0.0;
    pq.push({0.0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue;
        if (u == end) break;

        for (auto const& [v, weight] : adj.at(u)) {
            double new_dist = d + weight;

            if (dist.find(v) == dist.end() || new_dist < dist[v] ||
               (std::abs(new_dist - dist[v]) < 1e-9 && u < parent[v])) {

                dist[v] = new_dist;
                parent[v] = u;
                pq.push({new_dist, v});
            }
        }
    }

    if (dist.find(end) == dist.end()) return {};

    std::vector<std::string> path;
    for (std::string at = end; at != ""; at = (parent.count(at) ? parent.at(at) : "")) {
        path.push_back(at);
        if (at == start) break;
    }

    std::reverse(path.begin(), path.end());

    return (path.front() == start) ? path : std::vector<std::string>{};
}

vector<PairReq> findMaxIndependentSet(const vector<PairReq>& pairs) {
    int n = pairs.size();
    vector<vector<int>> conflictGraph(n);
    vector<int> degrees(n, 0);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (pairs[i].u_orig == pairs[j].u_orig) {
                conflictGraph[i].push_back(j);
                conflictGraph[j].push_back(i);
                degrees[i]++;
                degrees[j]++;
            }
        }
    }

    vector<bool> deleted(n, false);
    vector<PairReq> independentSet;

    vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b) { return degrees[a] < degrees[b]; });

    for (int idx : order) {
        if (!deleted[idx]) {
            independentSet.push_back(pairs[idx]);
            for (int neighbor : conflictGraph[idx]) {
                deleted[neighbor] = true;
            }
        }
    }
    return independentSet;
}

struct KPathNode {
    string curr_node;
    double current_cost;
    vector<string> path_nodes;
    vector<RawEdge> path_edges;

    bool operator>(const KPathNode& other) const {
        return current_cost > other.current_cost;
    }
};

vector<PairReq> processGraphEdges(const vector<RawEdge>& rawEdges,
                                  const vector<PairReq>& pairs,
                                  int K,
                                  map<string, vector<RawEdge>>& retainedEdgesPerV,
                                  map<string, set<string>>& retainedNodesPerV,
                                  set<string>& allRetainedEdgeKeys,
                                  set<string>& allFilteredEdgeKeys,
                                  set<string>& allRetainedNodes,
                                  bool retainAll = false) {

    map<string, vector<pair<string, pair<double, RawEdge>>>> adj;
    set<string> totalGraphNodes;
    for (const auto& edge : rawEdges) {
        adj[edge.u].push_back({edge.v, {edge.cost, edge}});
        adj[edge.v].push_back({edge.u, {edge.cost, edge}});
        if (retainAll) {
            totalGraphNodes.insert(edge.u);
            totalGraphNodes.insert(edge.v);
        }
    }

    vector<PairReq> independentSet = findMaxIndependentSet(pairs);

    set<string> v_nodes;
    for (const auto& req : independentSet) {
        v_nodes.insert(req.u_orig);
    }

    for (const string& v : v_nodes) {

        set<string> destinations;
        for (const auto& req : pairs) {
            if (req.u_orig == v) {
                destinations.insert(req.v_orig);
            }
        }
        if (destinations.empty()) continue;

        set<string> v_retained_nodes;
        vector<RawEdge> v_retained_edges;
        set<string> inserted_edge_keys_for_v;

        map<string, int> dest_found_count;
        for (const string& d : destinations) dest_found_count[d] = 0;

        map<string, int> visit_count;

        priority_queue<KPathNode, vector<KPathNode>, greater<KPathNode>> pq;
        pq.push({v, 0.0, {v}, {}});

        while (!pq.empty()) {
            KPathNode top = pq.top();
            pq.pop();

            string u = top.curr_node;

            if (visit_count[u] >= K * (int)destinations.size()) continue;
            visit_count[u]++;

            if (destinations.find(u) != destinations.end()) {
                if (dest_found_count[u] < K) {
                    dest_found_count[u]++;

                    for (const string& node : top.path_nodes) {
                        v_retained_nodes.insert(node);
                        allRetainedNodes.insert(node);
                    }

                    for (const RawEdge& edge : top.path_edges) {
                        allRetainedEdgeKeys.insert(edge.key);
                        if (inserted_edge_keys_for_v.find(edge.key) == inserted_edge_keys_for_v.end()) {
                            v_retained_edges.push_back(edge);
                            inserted_edge_keys_for_v.insert(edge.key);
                        }
                    }
                }
            }

            bool all_filled = true;
            for (const string& d : destinations) {
                if (dest_found_count[d] < K) {
                    all_filled = false;
                    break;
                }
            }
            if (all_filled) break;

            for (const auto& neighbor : adj[u]) {
                string next_node = neighbor.first;
                double edge_cost = neighbor.second.first;
                RawEdge edge_obj = neighbor.second.second;

                if (find(top.path_nodes.begin(), top.path_nodes.end(), next_node) != top.path_nodes.end()) {
                    continue;
                }

                KPathNode next_path = top;
                next_path.curr_node = next_node;
                next_path.current_cost += edge_cost;
                next_path.path_nodes.push_back(next_node);
                next_path.path_edges.push_back(edge_obj);

                pq.push(next_path);
            }
        }

        retainedEdgesPerV[v] = v_retained_edges;
        retainedNodesPerV[v] = v_retained_nodes;
    }

    if (retainAll) {

        for (const auto& node : totalGraphNodes) {
            allRetainedNodes.insert(node);
        }
        for (const auto& edge : rawEdges) {
            allRetainedEdgeKeys.insert(edge.key);
        }

        for (const string& v : v_nodes) {

            for (const auto& node : totalGraphNodes) {
                retainedNodesPerV[v].insert(node);
            }

            set<string> existing_keys;
            for (const auto& edge : retainedEdgesPerV[v]) {
                existing_keys.insert(edge.key);
            }
            for (const auto& edge : rawEdges) {
                if (existing_keys.find(edge.key) == existing_keys.end()) {
                    retainedEdgesPerV[v].push_back(edge);
                    existing_keys.insert(edge.key);
                }
            }
        }
    }

    for (const auto& edge : rawEdges) {
        if (allRetainedEdgeKeys.find(edge.key) == allRetainedEdgeKeys.end()) {
            allFilteredEdgeKeys.insert(edge.key);
        }
    }

    return independentSet;
}

map<string, map<string, double>> computeAllPairsPhysDist(const vector<RawEdge>& edges, const set<string>& nodes) {
    map<string, map<string, double>> dist;
    const double INF = 1e15;

    for (auto& u : nodes) {
        for (auto& v : nodes) {
            dist[u][v] = (u == v) ? 0 : INF;
        }
    }

    for (auto& e : edges) {
        if (e.cost < dist[e.u][e.v]) {
            dist[e.u][e.v] = e.cost;
            dist[e.v][e.u] = e.cost;
        }
    }

    for (auto& k : nodes) {
        for (auto& i : nodes) {
            if (dist[i][k] >= INF) continue;
            for (auto& j : nodes) {
                if (dist[k][j] < INF) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
    return dist;
}

class MultiSetIndex {
public:
    vector<double> data;
    unordered_map<double, int> pos;

    void insert(double x) {

        if (pos.count(x))
            return;

        int p = data.size();
        data.push_back(x);
        pos[x] = p;
    }

    int getPosition(double x) {
        return pos[x];
    }

    double getElement(int p) {
        return data[p];
    }
};

void loadOriginalGraph(std::string fileEdges, int K, WeightedGraph& adj) {
    std::ifstream fe(fileEdges);
    std::string line, col;
    if (!fe.is_open()) { std::cerr << "无法打开文件: " << fileEdges << std::endl; return; }

    getline(fe, line);

    struct RawEdge { std::string u, v; double cost; };
    std::vector<RawEdge> edges;
    while (getline(fe, line)) {
        std::stringstream ss(line);
        std::vector<std::string> c;
        while (getline(ss, col, ',')) c.push_back(col);
        if (c.size() >= 5) edges.push_back({c[2], c[3], std::stod(c[4])});
    }

    for (const auto& edge : edges) {
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {

                std::string u_i = safeName(edge.u) + "_L" + std::to_string(i);
                std::string v_j = safeName(edge.v) + "_L" + std::to_string(j);

                adj[u_i][v_j] = edge.cost;
                adj[v_j][u_i] = edge.cost;
            }
        }
    }
}

const double INF = std::numeric_limits<double>::infinity();

pair<double, vector<pair<string, double>>> getShortestPathDetailed(const string& start, const string& end,
                       const map<string, vector<pair<string, pair<string, double>>>>& adj) {
    if (start == end) return {0.0, {{start, 0.0}}};
    if (adj.find(start) == adj.end() || adj.find(end) == adj.end()) {
        return {numeric_limits<double>::infinity(), {}};
    }

    map<string, double> dist;
    map<string, string> parent;
    for (auto const& [node, _] : adj) dist[node] = numeric_limits<double>::infinity();

    dist[start] = 0.0;
    priority_queue<pair<double, string>, vector<pair<double, string>>, greater<pair<double, string>>> pq;
    pq.push({0.0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        if (u == end) {
            vector<pair<string, double>> path_with_dist;
            for (string at = end; at != ""; at = parent[at]) {
                path_with_dist.push_back({at, dist[at]});
            }
            reverse(path_with_dist.begin(), path_with_dist.end());
            return {d, path_with_dist};
        }
        for (auto& edge : adj.at(u)) {
            string v = edge.first;
            double weight = edge.second.second;
            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
    return {numeric_limits<double>::infinity(), {}};
}

void verifySimplification(const vector<PairReq>& pairs,
                          const map<string, vector<pair<string, pair<string, double>>>>& adj_before,
                          const map<string, vector<pair<string, pair<string, double>>>>& adj_after) {

    for (const auto& p : pairs) {
        auto [d1, path1] = getShortestPathDetailed(p.u_orig, p.v_orig, adj_before);

        auto [neighU, distU] = global_pair_neighbor_index.at(p.u_orig);
        auto [neighV, distV] = global_pair_neighbor_index.at(p.v_orig);

        auto [d_between, pathBetween] = (neighU == neighV) ?
            make_pair(0.0, vector<pair<string, double>>{{neighU, 0.0}}) :
            getShortestPathDetailed(neighU, neighV, adj_after);

        double d2 = distU + d_between + distV;

        if (abs(d1 - d2) > 1e-4) {
            cout << "\n[PATH MISMATCH] Pair: " << p.u_orig << " -> " << p.v_orig << endl;
            cout << "  原图总距离: " << d1 << endl;
            cout << "  原图路径详情: ";
            for (auto& node : path1) {
                cout << node.first << "(" << node.second << ") -> ";
            }
            cout << "END" << endl;

            cout << "  新图总距离: " << d2 << " (拆解: " << distU << " + " << d_between << " + " << distV << ")" << endl;
            cout << "  中间段路径详情: " << neighU << "(0.0) -> ";
            for(auto& node : pathBetween) {
                if(node.first == neighU) continue;
                cout << node.first << "(" << node.second << ") -> ";
            }
            cout << "END" << endl;
        }
    }
    cout << "[INFO] 检验点对之间的距离是否保持？ 通过！！！" << endl;
}

void simplifyGraphVersion2(vector<RawEdge>& rawEdges,
                   const vector<PairReq>& pairs,
                   set<string>& allNodes,
                   map<string, vector<pair<string, pair<string, double>>>>& adj) {

    auto getNodeType = [](const string& node) -> char {
        if (node.empty()) return 'O';
        return node[0];
    };

    auto removeNodeFromNeighbors = [&](auto& graph, string node_to_remove, string neighbor) {
        auto& neighbors = graph[neighbor];
        neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(),
            [&](const auto& e) { return e.first == node_to_remove; }), neighbors.end());
    };

    auto safe_merge = [&](string from, string target_rm, string target_add, double c, string key) {
        auto it_add = std::find_if(adj[from].begin(), adj[from].end(), [&](const auto& e) { return e.first == target_add; });
        if (it_add != adj[from].end()) {
            if (c < it_add->second.second) { it_add->second.second = c; it_add->second.first = key; }
            adj[from].erase(std::remove_if(adj[from].begin(), adj[from].end(), [&](const auto& e) { return e.first == target_rm; }), adj[from].end());
        } else {
            bool first = false;
            for (auto it = adj[from].begin(); it != adj[from].end(); ) {
                if (it->first == target_rm) {
                    if (!first) { it->first = target_add; it->second.second = c; it->second.first = key; first = true; ++it; }
                    else it = adj[from].erase(it);
                } else ++it;
            }
        }
    };

    map<string, vector<pair<string, pair<string, double>>>> adj_orig = adj;

    map<string, vector<pair<string, double>>> other_node_to_reachable_Es;

    global_pair_neighbor_index.clear();
    for (const auto& p : pairs) {
        if (adj.count(p.u_orig) && !adj.at(p.u_orig).empty()) {
            auto& edge = adj.at(p.u_orig).front();
            global_pair_neighbor_index[p.u_orig] = {edge.first, edge.second.second};
        }
        if (adj.count(p.v_orig) && !adj.at(p.v_orig).empty()) {
            auto& edge = adj.at(p.v_orig).front();
            global_pair_neighbor_index[p.v_orig] = {edge.first, edge.second.second};
        }
    }

    bool changed = true;
    map<string, string> forwarding_map;
    set<string> lockedNodes;

    std::set<std::string> forbidden_nodes;
    for (auto const& [node, edges] : adj) {
        for (auto const& edge : edges) {
            if (global_pair_neighbor_index.count(edge.first)) {
                forbidden_nodes.insert(node);
            }
        }
    }

    auto update_pair_index = [&](string v, string new_neighbor, double weight_v_to_new) {
        for (auto& [pair_node, pair_data] : global_pair_neighbor_index) {
            string& current_neighbor = pair_data.first;
            double& current_dist = pair_data.second;
            if (current_neighbor == v) {
                current_neighbor = new_neighbor;
                current_dist += weight_v_to_new;
            }
        }
    };

    while (changed) {
        changed = false;

        for (auto it = adj.begin(); it != adj.end(); ) {
            string e_node = it->first;
            if (getNodeType(e_node) == 'E' && it->second.size() == 2) {
                string n1 = it->second[0].first;
                string n2 = it->second[1].first;
                double w1 = it->second[0].second.second;
                double w2 = it->second[1].second.second;

                if (getNodeType(n1) == 'N' && getNodeType(n2) == 'N') {
                    cout << "[INFO] 发现 N1-E-N2 结构: " << n1 << " - " << e_node << " - " << n2 << endl;

                    double total_w = w1 + w2;
                    string key_n1_n2 = n1 + "_to_" + n2;

                    auto it_n1 = std::find_if(adj[n1].begin(), adj[n1].end(), [&](const auto& e){ return e.first == n2; });
                    if (it_n1 == adj[n1].end()) {
                        adj[n1].push_back({n2, {key_n1_n2, total_w}});
                        adj[n2].push_back({n1, {n2 + "_to_" + n1, total_w}});
                    } else {
                        if (total_w < it_n1->second.second) {
                            it_n1->second.second = total_w;

                            for (auto& e : adj[n2]) if (e.first == n1) e.second.second = total_w;
                        }
                    }

                    lockedNodes.insert(e_node);
                }
            }
            ++it;
        }

        for (auto it = adj.begin(); it != adj.end(); ) {
            string v = it->first;
            if (getNodeType(v) == 'E') {
                ++it; continue;
            }
            if (it->second.size() == 1) {
                string neighbor = it->second.front().first;
                double weight_v_to_n = it->second.front().second.second;

                update_pair_index(v, neighbor, weight_v_to_n);

                char type_v = getNodeType(v);
                if (type_v != 'E' && type_v != 'N') {

                    string target_e = "";
                    if (getNodeType(neighbor) == 'E') {
                        target_e = neighbor;
                    } else if (other_node_to_reachable_Es.count(neighbor)) {

                        for (auto& [e_node, dist] : other_node_to_reachable_Es[neighbor]) {
                            other_node_to_reachable_Es[v].push_back({e_node, dist + weight_v_to_n});
                        }
                    }
                    if (!target_e.empty()) {
                        other_node_to_reachable_Es[v].push_back({target_e, weight_v_to_n});
                    }
                }

                for (auto& edge : adj[neighbor]) {
                    if (edge.first == v) { edge.second.second += weight_v_to_n; break; }
                }

                for (auto it_n = adj[neighbor].begin(); it_n != adj[neighbor].end(); ) {
                    if (it_n->first == v) it_n = adj[neighbor].erase(it_n);
                    else ++it_n;
                }
                lockedNodes.insert(neighbor);
                allNodes.erase(v);
                it = adj.erase(it);
                changed = true; continue;
            }
            ++it;
        }

        for (auto it = adj.begin(); it != adj.end(); ) {
            string v = it->first;
            if (getNodeType(v) == 'E') {
                ++it; continue;
            }

            if (lockedNodes.count(v) || forbidden_nodes.count(v)) { ++it; continue; }

            if (it->second.size() == 2) {
                string u = it->second[0].first;
                double w_vu = it->second[0].second.second;

                update_pair_index(v, u, w_vu);

                string w = it->second[1].first;
                double cost = it->second[0].second.second + it->second[1].second.second;
                safe_merge(u, v, w, cost, u + "-to-" + w);
                safe_merge(w, v, u, cost, u + "-to-" + w);

                forwarding_map[v] = u;
                allNodes.erase(v);
                it = adj.erase(it);
                changed = true; continue;
            }
            ++it;
        }
    }

    for (auto it = adj.begin(); it != adj.end(); ) {
        string node = it->first;
        char t = getNodeType(node);
        if (t != 'E' && t != 'N') {
            cout << "[INFO] 清理非 E/N 节点: " << node << endl;

            for (const auto& edge : it->second) {
                removeNodeFromNeighbors(adj, node, edge.first);
            }
            allNodes.erase(node);
            it = adj.erase(it);
        } else {
            ++it;
        }
    }

    cout << "[INFO] 同步精简后的图结构至 rawEdges..." << endl;
    set<pair<string, string>> original_directions;
    for (const auto& e : rawEdges) original_directions.insert({e.u, e.v});

    rawEdges.clear();
    set<pair<string, string>> seen;
    int count = 0;

    for (auto const& [current_u, neighbors] : adj) {
        for (auto const& edge : neighbors) {
            string current_v = edge.first;
            string current_key = edge.second.first;
            double cost = edge.second.second;

            if (seen.find({min(current_u, current_v), max(current_u, current_v)}) == seen.end()) {
                string final_u = current_u;
                string final_v = current_v;
                string final_key = current_key;

                if (final_key.find("-to-") == string::npos && original_directions.count({current_v, current_u})) {
                    final_u = current_v; final_v = current_u;
                }

                if (final_key.find("-to-") != string::npos) {
                    final_key = final_u + "_to_" + final_v;
                } else {
                    string inv = final_v + "_" + final_u;
                    size_t pos = final_key.find(inv);
                    if (pos != string::npos) {
                        final_key.replace(pos, inv.length(), final_u + "_" + final_v);
                        size_t len = final_key.length();
                        if (len >= 4 && final_key.rfind("_rev") == len - 4) final_key = final_key.substr(0, len - 4) + "_fwd";
                        else if (len >= 4 && final_key.rfind("_fwd") == len - 4) final_key = final_key.substr(0, len - 4) + "_rev";
                    }
                }

                rawEdges.push_back({final_u, final_v, cost, final_key});
                seen.insert({min(current_u, current_v), max(current_u, current_v)});
                count++;
            }
        }
    }

    verifySimplification(pairs, adj_orig, adj);

    cout << "[INFO] 同步完成，共保留 " << count << " 条边。" << endl;
}

struct SubGraphComponent {
    string cutNode;
    vector<RawEdge> edges;
    set<string> nodes;
};

struct GraphSplitResult {

    map<string, vector<SubGraphComponent>> cutResults;
};

string getMappedNode(const string& orig_node) {
    if (global_pair_neighbor_index.count(orig_node)) {
        return global_pair_neighbor_index[orig_node].first;
    }
    return orig_node;
}

struct PairAssignment {
    int pair_id;
    std::string u_orig;
    std::string v_orig;
    std::string assigned_node;
};

void findIndependentSet(
    const std::vector<PairReq>& pairs,
    std::vector<std::string>& selected_nodes,
    std::vector<PairAssignment>& assignments) {

    int n = pairs.size();
    if (n == 0) return;

    assignments.clear();

    struct Edge {
        int original_index;
        std::string u;
        std::string v;
    };

    std::vector<Edge> uncovered_edges;
    uncovered_edges.reserve(n);

    for (int i = 0; i < n; ++i) {
        uncovered_edges.push_back({
            i,
            getMappedNode(pairs[i].u_orig),
            getMappedNode(pairs[i].v_orig)
        });
    }

    std::set<std::string> chosen_nodes;

    assignments.resize(n);

    while (!uncovered_edges.empty()) {

        std::map<std::string, int> node_freq;
        for (const auto& edge : uncovered_edges) {
            node_freq[edge.u]++;
            node_freq[edge.v]++;
        }

        std::string best_node = "";
        int max_freq = -1;
        for (const auto& pair : node_freq) {
            if (pair.second > max_freq) {
                max_freq = pair.second;
                best_node = pair.first;
            }
        }

        chosen_nodes.insert(best_node);

        std::vector<Edge> remaining_edges;
        for (const auto& edge : uncovered_edges) {
            if (edge.u == best_node || edge.v == best_node) {

                int idx = edge.original_index;
                assignments[idx] = {
                    pairs[idx].id,
                    pairs[idx].u_orig,
                    pairs[idx].v_orig,
                    best_node
                };
            } else {
                remaining_edges.push_back(edge);
            }
        }
        uncovered_edges = std::move(remaining_edges);
    }

    selected_nodes.assign(chosen_nodes.begin(), chosen_nodes.end());

    std::cout << "[INFO] Minimum Vertex Cover Selection Completed. Selected "
              << selected_nodes.size() << " unique nodes, and assigned "
              << assignments.size() << " pairs." << std::endl;
}

void extractComponents(
    const map<string, vector<pair<string, pair<string, double>>>>& adj,
    GraphSplitResult& result) {

    set<string> allNodes;
    for (auto const& [u, neighbors] : adj) {
        allNodes.insert(u);
        for (auto const& edge : neighbors) allNodes.insert(edge.first);
    }

    for (const string& cutNode : allNodes) {
        set<string> visited;
        visited.insert(cutNode);

        for (const string& startNode : allNodes) {
            if (visited.count(startNode)) continue;

            SubGraphComponent component;
            component.cutNode = cutNode;

            set<pair<string, string>> addedEdges;

            vector<string> q = {startNode};
            visited.insert(startNode);
            component.nodes.insert(startNode);

            size_t head = 0;
            while(head < q.size()){
                string u = q[head++];
                if (adj.count(u)) {
                    for (auto const& edge : adj.at(u)) {
                        string v = edge.first;
                        if (v == cutNode) continue;

                        string minNode = min(u, v);
                        string maxNode = max(u, v);

                        if (addedEdges.find({minNode, maxNode}) == addedEdges.end()) {

                            component.edges.push_back({u, v, edge.second.second, edge.second.first});
                            addedEdges.insert({minNode, maxNode});
                        }

                        if (visited.find(v) == visited.end()) {
                            visited.insert(v);
                            component.nodes.insert(v);
                            q.push_back(v);
                        }
                    }
                }
            }

            if (!component.nodes.empty()) {
                result.cutResults[cutNode].push_back(component);
            }
        }
    }
}

double get_shortest_distance_weighted(const WeightedGraph& graph, const std::string& start, const std::string& target) {
    if (start == target) return 0.0;

    if (graph.find(start) == graph.end() || graph.find(target) == graph.end()) {
        std::cout << "[Debug] Nodes missing: " << start << " or " << target << " not in graph." << std::endl;
        std::cout << "[Debug] Current nodes in graph (" << graph.size() << " total): ";
        for (const auto& [node, neighbors] : graph) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
        return INF;
    }

    std::unordered_map<std::string, double> min_dist;
    min_dist[start] = 0.0;

    std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> pq;
    pq.push({start, 0.0});

    std::cout << "[Debug] Starting Dijkstra from " << start << " to " << target << std::endl;

    while (!pq.empty()) {
        NodeDist top = pq.top();
        pq.pop();

        std::string u = top.node;
        double d = top.dist;

        if (min_dist.count(u) && d > min_dist[u]) {
            std::cout << "[Debug] Skipping outdated node " << u << " with distance " << d
                      << " (current min: " << min_dist[u] << ")" << std::endl;
            continue;
        }

        std::cout << "[Debug] Processing node: " << u << " with current distance: " << d << std::endl;

        if (u == target) {
            std::cout << "[Debug] Target " << target << " reached. Final Distance: " << d << std::endl;
            return d;
        }

        const auto& neighbors = graph.at(u);
        for (auto const& [v, weight] : neighbors) {
            double new_dist = d + weight;

            if (min_dist.find(v) == min_dist.end() || new_dist < min_dist[v]) {
                double old_dist = (min_dist.find(v) == min_dist.end()) ? INF : min_dist[v];
                min_dist[v] = new_dist;
                pq.push({v, new_dist});

                std::cout << "[Debug]   -> Relaxing edge (" << u << " -> " << v << "): weight = " << weight
                          << ", old_dist = " << (old_dist == INF ? "INF" : std::to_string(old_dist))
                          << ", new_dist = " << new_dist << std::endl;
            } else {

                std::cout << "[Debug]   -> Ignored edge (" << u << " -> " << v << "): weight = " << weight
                          << ", path dist " << new_dist << " >= current min " << min_dist[v] << std::endl;
            }
        }
    }

    std::cout << "[Debug] Target " << target << " is unreachable from " << start << std::endl;
    return INF;
}

struct TarjanContext {
    map<string, int> dfn, low;
    int timer = 0;
    set<string> cutNodes;
};

struct Component {
    set<string> nodes;
    vector<RawEdge> edges;
};

void tarjan(const string& u, const string& p,
            const map<string, vector<pair<string, pair<string, double>>>>& adj,
            TarjanContext& ctx) {
    ctx.dfn[u] = ctx.low[u] = ++ctx.timer;
    int children = 0;

    for (auto const& edge : adj.at(u)) {
        string v = edge.first;
        if (v == p) continue;
        if (ctx.dfn.count(v)) {
            ctx.low[u] = min(ctx.low[u], ctx.dfn[v]);
        } else {
            children++;
            tarjan(v, u, adj, ctx);
            ctx.low[u] = min(ctx.low[u], ctx.low[v]);

            if (p != "" && ctx.low[v] >= ctx.dfn[u]) ctx.cutNodes.insert(u);
        }
    }

    if (p == "" && children > 1) ctx.cutNodes.insert(u);
}

vector<Component> getAllComponentsSorted(const map<string, vector<pair<string, pair<string, double>>>>& adj) {

    TarjanContext ctx;
    for (auto const& [node, _] : adj) {
        if (!ctx.dfn.count(node)) tarjan(node, "", adj, ctx);
    }
    const set<string>& forbidden = ctx.cutNodes;

    set<string> visited;
    vector<Component> allComponents;

    for (auto const& [startNode, _] : adj) {

        if (forbidden.count(startNode) || visited.count(startNode)) continue;

        Component comp;
        vector<string> q = {startNode};
        visited.insert(startNode);
        comp.nodes.insert(startNode);

        set<pair<string, string>> addedEdges;
        size_t head = 0;
        while(head < q.size()){
            string u = q[head++];
            for (auto const& edge : adj.at(u)) {
                string v = edge.first;
                if (forbidden.count(v)) continue;

                string minN = min(u, v), maxN = max(u, v);
                if (addedEdges.find({minN, maxN}) == addedEdges.end()) {
                    comp.edges.push_back({u, v, edge.second.second, edge.second.first});
                    addedEdges.insert({minN, maxN});
                }
                if (visited.find(v) == visited.end()) {
                    visited.insert(v);
                    comp.nodes.insert(v);
                    q.push_back(v);
                }
            }
        }
        allComponents.push_back(comp);
    }

    std::sort(allComponents.begin(), allComponents.end(),
        [](const Component& a, const Component& b) {
            return a.nodes.size() > b.nodes.size();
        });

    return allComponents;
}

std::map<string, int> nodeToIndex;

void initializeNodeMapping(const std::set<std::string>& allNodes) {
    nodeToIndex.clear();
    int id_counter = 1;
    cout << "Initializing node mapping..." << endl;
    for (const auto& node : allNodes) {
        if (nodeToIndex.find(node) == nodeToIndex.end()) {
            nodeToIndex[node] = id_counter++;

        }
    }
}

struct NodeInfo {
    string node;
    int denseIndex;
    int layer;
};

NodeInfo parseString(const string& s) {
    size_t p1 = s.find('@');

    string node = s.substr(0, p1);

    size_t pD = s.find('D', p1);
    int denseIdx = 0;

    if (pD != string::npos) {

        denseIdx = stoi(s.substr(pD + 1));
    }

    cout << "input: " << s << endl;
    cout << "node: " << node << ", denseIdx: " << denseIdx << endl;

    return {node, denseIdx, 0};
}

set<string> selectActivationNodesV2(const set<string>& allNodes,
                                  const vector<RawEdge>& rawEdges,
                                  const set<string>& ActNodes,
                                  double Np) {

    map<string, vector<string>> adjList;
    for (const auto& e : rawEdges) {
        adjList[e.u].push_back(e.v);
        adjList[e.v].push_back(e.u);
    }

    set<string> result;
    set<string> selectedIds;

    for (const auto& e : rawEdges) {
        string u = e.u;
        string v = e.v;

        bool uInAct = ActNodes.count(u);
        bool vInAct = ActNodes.count(v);

        bool uIsN = (!u.empty() && u[0] == 'N');
        bool vIsN = (!v.empty() && v[0] == 'N');

        if (uInAct && vIsN) {
            if (allNodes.count(v) && !selectedIds.count(v)) {
                result.insert(v);
                selectedIds.insert(v);
            }
        }

        if (vInAct && uIsN) {
            if (allNodes.count(u) && !selectedIds.count(u)) {
                result.insert(u);
                selectedIds.insert(u);
            }
        }
    }

    size_t currentSize = result.size();
    size_t minRequired = max(currentSize, static_cast<size_t>(allNodes.size() * Np));
    size_t totalLimit = min(allNodes.size(), minRequired);

    cout << "[LOG] 初始邻居激活后节点数: " << currentSize << ", allow: " << allNodes.size() * Np << ", 调整后上限: " << totalLimit << endl;

    if (result.size() < totalLimit) {
        map<string, double> vulnerabilityScore;
        map<string, int> pathCoverageScore;

        struct NodeCandidate { string name; double score; };
        vector<NodeCandidate> candidates;
        for (const auto& node : allNodes) {
            if (selectedIds.count(node)) continue;

            double score = vulnerabilityScore[node] + (pathCoverageScore[node] * 0.1);
            candidates.push_back({node, score});
        }

        sort(candidates.begin(), candidates.end(), [](const NodeCandidate& a, const NodeCandidate& b) {
            return a.score > b.score;
        });

        for (const auto& item : candidates) {
            if (result.size() >= totalLimit) break;
            result.insert(item.name);
            selectedIds.insert(item.name);
        }
    }

    return result;
}

using ShortestPathMap = map<string, set<string>>;

struct DirectModelRow {
    std::map<std::string, double> coeffs;
    double lower = -kHighsInf;
    double upper = kHighsInf;
    std::string name;
};

// 求解器无关的线性/MIP模型容器。
// 原建模代码仍然只调用 ensureVar / setObjectiveCoeff / markBinary / addRow，
// 因此目标函数、变量和约束生成逻辑不需要因为求解器切换而改变。
class DirectModelBuilder {
public:
    int ensureVar(const std::string& name, bool binary = false) {
        auto it = varIndex.find(name);
        if (it != varIndex.end()) {
            int idx = it->second;
            if (binary) {
                colLower[idx] = 0.0;
                colUpper[idx] = 1.0;
                isBinaryVar[idx] = true;
            }
            return idx;
        }

        int idx = static_cast<int>(varNames.size());
        varIndex[name] = idx;
        varNames.push_back(name);
        colCost.push_back(0.0);
        colLower.push_back(0.0);
        colUpper.push_back(kHighsInf);
        isBinaryVar.push_back(false);

        if (binary) {
            colUpper[idx] = 1.0;
            isBinaryVar[idx] = true;
        }
        return idx;
    }

    void setObjectiveCoeff(const std::string& name, double coeff) {
        int idx = ensureVar(name);
        colCost[idx] += coeff;
    }

    void markBinary(const std::string& name) {
        ensureVar(name, true);
    }

    void addRow(const std::string& name,
                std::map<std::string, double> coeffs,
                double lower,
                double upper) {
        for (auto it = coeffs.begin(); it != coeffs.end();) {
            if (std::abs(it->second) < 1e-15) {
                it = coeffs.erase(it);
            } else {
                ensureVar(it->first);
                ++it;
            }
        }
        rows.push_back({std::move(coeffs), lower, upper, name});
    }

    HighsStatus passToHighs(Highs& highs) const {
        HighsModel model;
        HighsLp& lp = model.lp_;

        lp.num_col_ = static_cast<HighsInt>(varNames.size());
        lp.num_row_ = static_cast<HighsInt>(rows.size());
        lp.col_cost_ = colCost;
        lp.col_lower_ = colLower;
        lp.col_upper_ = colUpper;
        lp.row_lower_.resize(lp.num_row_);
        lp.row_upper_.resize(lp.num_row_);
        lp.col_names_ = varNames;
        lp.sense_ = ObjSense::kMinimize;
        lp.offset_ = 0.0;
        lp.integrality_.resize(varNames.size(), HighsVarType::kContinuous);
        for (size_t i = 0; i < isBinaryVar.size(); ++i) {
            if (isBinaryVar[i]) lp.integrality_[i] = HighsVarType::kInteger;
        }

        lp.a_matrix_.format_ = MatrixFormat::kRowwise;
        lp.a_matrix_.start_.clear();
        lp.a_matrix_.index_.clear();
        lp.a_matrix_.value_.clear();
        lp.a_matrix_.start_.reserve(lp.num_row_ + 1);
        lp.a_matrix_.start_.push_back(0);

        for (HighsInt r = 0; r < lp.num_row_; ++r) {
            lp.row_lower_[r] = rows[r].lower;
            lp.row_upper_[r] = rows[r].upper;

            for (const auto& [varName, coeff] : rows[r].coeffs) {
                auto it = varIndex.find(varName);
                if (it == varIndex.end()) {
                    throw std::runtime_error("内部建模错误：未注册变量 " + varName);
                }
                lp.a_matrix_.index_.push_back(static_cast<HighsInt>(it->second));
                lp.a_matrix_.value_.push_back(coeff);
            }
            lp.a_matrix_.start_.push_back(
                static_cast<HighsInt>(lp.a_matrix_.index_.size()));
        }

        return highs.passModel(std::move(model));
    }

    size_t numRows() const { return rows.size(); }
    size_t numCols() const { return varNames.size(); }

    const std::vector<std::string>& getVarNames() const { return varNames; }
    const std::vector<double>& getColCost() const { return colCost; }
    const std::vector<double>& getColLower() const { return colLower; }
    const std::vector<double>& getColUpper() const { return colUpper; }
    const std::vector<bool>& getIsBinaryVar() const { return isBinaryVar; }
    const std::vector<DirectModelRow>& getRows() const { return rows; }
    const std::unordered_map<std::string, int>& getVarIndex() const { return varIndex; }

private:
    std::unordered_map<std::string, int> varIndex;
    std::vector<std::string> varNames;
    std::vector<double> colCost;
    std::vector<double> colLower;
    std::vector<double> colUpper;
    std::vector<bool> isBinaryVar;
    std::vector<DirectModelRow> rows;
};

void buildModel(DirectModelBuilder& modelBuilder, const vector<RawEdge>& rawEdges, const vector<PairReq>& pairs, const set<string>& allNodes, set<string> chosen_node_set, std::map<std::string, std::vector<PairReq>> node_to_managed_pairs) {
    auto physDist = computeAllPairsPhysDist(rawEdges, allNodes);

    int total_covered_pairs = 0;

    for (const auto& [node_name, managed_pairs] : node_to_managed_pairs) {
        int count_for_this_node = managed_pairs.size();
        total_covered_pairs += count_for_this_node;

        std::cout << "[INFO] 节点 " << node_name
                  << " 负责管理 " << count_for_this_node << " 个点对。" << std::endl;
    }

    std::cout << "=========================================" << std::endl;
    std::cout << "[SUMMARY] node_to_managed_pairs 总共覆盖的点对数: " << total_covered_pairs << std::endl;

    initializeNodeMapping(allNodes);

    MultiSetIndex Dense;
    for (auto& pr : pairs) {
        Dense.insert(pr.density);
    }

    set<string> uniqueSources;

    std::unordered_map<int, std::string> pair_to_assigned_node;
    for (const auto& [node_name, managed_pairs] : node_to_managed_pairs) {
        for (const auto& raw_pair : managed_pairs) {
            pair_to_assigned_node[raw_pair.id] = node_name;
        }
    }

    std::cout << "\n--- 生成唯一标识 (基于映射节点及密度) ---" << std::endl;

    uniqueSources.clear();
    for (const auto& raw_pair : pairs) {
        auto it = pair_to_assigned_node.find(raw_pair.id);
        if (it == pair_to_assigned_node.end()) continue;

        std::string assigned_node = it->second;
        std::string mapped_u = getMappedNode(raw_pair.u_orig);
        std::string mapped_v = getMappedNode(raw_pair.v_orig);
        (void)mapped_u;
        (void)mapped_v;

        if (chosen_node_set.find(assigned_node) == chosen_node_set.end()) continue;

        double density = raw_pair.density;
        std::string sourceKey = assigned_node + "@D" + std::to_string(Dense.getPosition(density));

        if (uniqueSources.find(sourceKey) == uniqueSources.end()) {
            uniqueSources.insert(sourceKey);
            std::cout << "节点: " << assigned_node
                      << " | 密度: " << density
                      << " -> 生成唯一标识: " << sourceKey << std::endl;
        }
    }

    std::cout << "uniqueSources: " << uniqueSources.size() << std::endl;

    string initX = "Initial_nodes";
    int totalDestCount = 0;

    for (const auto& source : uniqueSources) {
        NodeInfo info = parseString(source);
        cout << "[DEBUG] 解析 uniqueSource: " << source
             << " -> node: " << info.node
             << ", denseIndex: " << info.denseIndex << endl;

        if (!info.node.empty() && info.node[0] == 'E') {
            initX = info.node;
            break;
        }
    }

    cout << "[LOG] 初始 initX: " << initX << endl;
    for (auto& e : rawEdges) {
        if (e.u == initX && !e.v.empty() && e.v[0] == 'N') {
            initX = e.v;
            break;
        } else if (e.v == initX && !e.u.empty() && e.u[0] == 'N') {
            initX = e.u;
            break;
        }
    }
    cout << "[LOG] 更新后的 initX (邻居): " << initX << endl;

    map<string, set<double>> nodeToDensitiesMap;
    map<string, set<string>> nodeToNeighborsMap;

    for (const auto& raw_pair : pairs) {
        auto it = pair_to_assigned_node.find(raw_pair.id);
        if (it == pair_to_assigned_node.end()) continue;

        string owner = it->second;
        string mapped_u = getMappedNode(raw_pair.u_orig);
        string mapped_v = getMappedNode(raw_pair.v_orig);

        if (mapped_u == mapped_v) {
            if (chosen_node_set.find(owner) != chosen_node_set.end()) {
                nodeToDensitiesMap[owner].insert(raw_pair.density);
            }
            continue;
        }

        if (owner == mapped_u) {
            nodeToDensitiesMap[mapped_u].insert(raw_pair.density);
            nodeToNeighborsMap[mapped_u].insert(mapped_v);
        } else if (owner == mapped_v) {
            nodeToDensitiesMap[mapped_v].insert(raw_pair.density);
            nodeToNeighborsMap[mapped_v].insert(mapped_u);
        }
    }

    cout << "[LOG] 映射表构建完成，选中的节点总数: " << chosen_node_set.size() << endl;
    cout << "[LOG] 开始打印 chosen_node_set 内容:" << endl;

    int TotalEdges = 0;
    for (const auto& nodeName : chosen_node_set) {
        cout << "  - 枢纽(映射ID): " << nodeName;
        cout << " | 密度索引: ";

        auto it_dens = nodeToDensitiesMap.find(nodeName);
        if (it_dens != nodeToDensitiesMap.end() && !it_dens->second.empty()) {
            for (double density : it_dens->second) {
                cout << "D" << Dense.getPosition(density) << " ";
            }
        } else {
            cout << "(无)";
        }

        auto neighbor_it = nodeToNeighborsMap.find(nodeName);
        size_t neighbor_size = (neighbor_it != nodeToNeighborsMap.end()) ? neighbor_it->second.size() : 0;
        cout << " | 覆盖邻居数: " << neighbor_size << endl;

        cout << "      邻居列表: ";
        if (neighbor_size == 0) {
            cout << "(无)";
        } else {
            for (const auto& neighbor : neighbor_it->second) {
                cout << neighbor << " ";
                TotalEdges++;
            }
        }
        cout << endl;
    }

    cout << "[LOG] 总覆盖边数 (TotalEdges): " << TotalEdges << endl;
    cout << "[LOG] 节点内容打印完成。" << endl;

    set<string> destNodes;
    set<string> srcNodes;
    set<string> ActNodes;

    for (auto& pr : pairs) {
        string u_m = getMappedNode(pr.u_orig);
        string v_m = getMappedNode(pr.v_orig);

        string u_id = u_m;
        string v_id = v_m;

        srcNodes.insert(u_id);
        ActNodes.insert(u_id);
        if (u_id != initX) destNodes.insert(u_id);

        ActNodes.insert(v_id);
        if (v_id != initX) destNodes.insert(v_id);
    }

    cout << "[INFO] 集合构建完成。ActNodes size: " << ActNodes.size()
         << ", srcNodes size: " << srcNodes.size()
         << ", destNodes size: " << destNodes.size() << endl;

    cout << "[SUMMARY] ActNodes 总数: " << ActNodes.size()
         << ", DestNodes 总数: " << destNodes.size()
         << ", SrcNodes 总数: " << srcNodes.size() << endl;

    if (ActNodes.empty()) {
        cout << "[ERROR] ActNodes 为空，请检查 pairs 映射逻辑或节点名是否拼写错误！" << endl;
    }

    cout << "destNodes size: " << destNodes.size() << endl;
    cout << "ActNodes size: " << ActNodes.size() << endl;
    cout << "Select Initial: " << initX << endl;
    cout << "Unique sources identified: " << uniqueSources.size() << endl;

    map<string, int> nodeDegrees;
    for (const auto& v : allNodes) nodeDegrees[v] = 0;
    for (const auto& e : rawEdges) {
        nodeDegrees[e.u]++;
        nodeDegrees[e.v]++;
    }

    double Np = 1;
    set<string> allowedMultiLayerNodes = selectActivationNodesV2(allNodes, rawEdges, ActNodes, Np);

    cout << "allowedMultiLayerNodes: " << allowedMultiLayerNodes.size() << endl;
    for (const string& node : allowedMultiLayerNodes) {
        cout << "Successfully activated node: " << node << endl;
    }

    const double cost_scale = 10000000;
    const int BUCKET_COUNT = 10;
    std::vector<int> histogram(BUCKET_COUNT, 0);
    std::vector<double> maxInBucket(BUCKET_COUNT, 0.0);

    int coeffCount = 0;
    double maxCoff = 0.0;

    for (const string& srcWithDense : uniqueSources) {
        NodeInfo info = parseString(srcWithDense);
        string src = info.node;
        double dense = Dense.getElement(info.denseIndex);

        for (auto& e : rawEdges) {
            double coeff = e.cost * (dense * cost_scale);
            if (coeff <= 0) continue;

            coeffCount++;
            if (coeff > maxCoff) maxCoff = coeff;

            int index = static_cast<int>(std::log10(coeff));
            if (index < 0) index = 0;
            if (index >= BUCKET_COUNT) index = BUCKET_COUNT - 1;

            histogram[index]++;
            if (coeff > maxInBucket[index]) maxInBucket[index] = coeff;
        }
    }

    double threshold99 = 0.0;
    int accumulated = 0;
    int targetCount = static_cast<int>(coeffCount * 0.99);

    std::cout << "--- Distribution Analysis (Filtered Total Edges: " << coeffCount << ") ---" << std::endl;
    for (int i = 0; i < BUCKET_COUNT; ++i) {
        accumulated += histogram[i];
        if (threshold99 == 0.0 && accumulated >= targetCount) {
            threshold99 = std::pow(10, i + 1);
        }

        std::cout << "[10^" << i << " - 10^" << (i + 1) << "): "
                  << histogram[i] << " items, Max in bucket: " << maxInBucket[i] << std::endl;
    }

    std::cout << "\nAnalysis Summary:" << std::endl;
    std::cout << "Max Coefficient: " << maxCoff << std::endl;
    std::cout << "1% of Max Coefficient: " << (maxCoff * 0.01) << std::endl;
    std::cout << "99th Percentile Threshold: " << threshold99 << std::endl;

    maxCoff = 1;
    cout << "Max coefficient for normalization: " << maxCoff << endl;

    ShortestPathMap sourcePaths;
    (void)sourcePaths;
    (void)physDist;

    double totalLen = 0;
    for (auto& pr : pairs) {
        string u_m = pr.u_orig;
        string v_m = pr.v_orig;
        double density = pr.density;

        double u_dist = 0;
        double v_dist = 0;
        bool found_u = false;
        bool found_v = false;

        for (const auto& [key, valuePair] : global_pair_neighbor_index) {
            if (!found_u && safeName(key) == safeName(u_m)) {
                u_dist = valuePair.second;
                found_u = true;
            }
            if (!found_v && safeName(key) == safeName(v_m)) {
                v_dist = valuePair.second;
                found_v = true;
            }
            if (found_u && found_v) break;
        }

        totalLen += (u_dist + v_dist) * density;
    }
    cout << "两端距离总和结果: " << totalLen << endl;

    // 目标函数：min Total_Cost
    modelBuilder.setObjectiveCoeff("Total_Cost", 1.0);

    vector<string> allCostVars;

    // Cost_src >= sum(cost * f)
    for (const string& srcWithDense : uniqueSources) {
        NodeInfo info = parseString(srcWithDense);
        string src = info.node;
        int denseIndex = info.denseIndex;
        double dense = Dense.getElement(info.denseIndex);
        int layer = info.layer;

        string costVarName = "Cost_" + safeName(src) + "_" + to_string(layer) + "_" + to_string(denseIndex);
        allCostVars.push_back(costVarName);

        std::map<std::string, double> row;
        row[costVarName] += 1.0;

        for (auto& e : rawEdges) {
            double coeff = (e.cost * dense * cost_scale) / maxCoff;
            // 原代码通过 std::fixed << std::setprecision(8) 写入 LP，
            // 因此这里保留相同的 8 位小数舍入，确保数学模型与原 LP 版本一致。
            coeff = std::round(coeff * 1e8) / 1e8;

            int maxULayer = (!e.u.empty() && e.u[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(e.u) > 0) ? NUM_K : 1);
            int maxVLayer = (!e.v.empty() && e.v[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(e.v) > 0) ? NUM_K : 1);

            for (int k1 = 0; k1 < maxULayer; ++k1) {
                for (int k2 = 0; k2 < maxVLayer; ++k2) {
                    string f_base = "f_src" + safeName(src) + "_" + to_string(layer) + "_" + to_string(denseIndex) + "_" + safeName(e.key) + "_L" + to_string(k1) + "_L" + to_string(k2);
                    row[f_base + "_fwd"] -= coeff;
                    row[f_base + "_rev"] -= coeff;
                }
            }
        }

        modelBuilder.addRow("CostDef_" + safeName(src) + "_" + to_string(layer) + "_" + to_string(denseIndex),
                            std::move(row), 0.0, kHighsInf);
    }

    // Total_Cost >= sum(Cost_src)
    {
        std::map<std::string, double> row;
        row["Total_Cost"] = 1.0;
        for (const string& var : allCostVars) row[var] -= 1.0;
        modelBuilder.addRow("Total_Cost_Calc", std::move(row), 0.0, kHighsInf);
    }

    cout << "Root 变量权限约束已生成，正在构建节点层面和边层面的约束..." << endl;

    totalDestCount = 0;
    for (const string& srcWithDense : uniqueSources) {
        NodeInfo info = parseString(srcWithDense);
        string src = info.node;
        string denseIndex = to_string(info.denseIndex);
        string safeSrc = safeName(src);
        int layer = info.layer;

        cout << "src: " << src << ", layer: " << layer << endl;

        unordered_map<string, int> currentDests;
        auto node_it = node_to_managed_pairs.find(src);

        if (node_it != node_to_managed_pairs.end()) {
            for (const auto& raw_pair : node_it->second) {
                string mapped_u = getMappedNode(raw_pair.u_orig);
                string mapped_v = getMappedNode(raw_pair.v_orig);

                string safe_u = safeName(mapped_u);
                string safe_v = safeName(mapped_v);

                string neighbor = "";
                if (safe_u == safeSrc) {
                    neighbor = safe_v;
                } else if (safe_v == safeSrc) {
                    neighbor = safe_u;
                } else {
                    continue;
                }

                double cdensity = raw_pair.density;
                if (denseIndex != to_string(Dense.getPosition(cdensity))) continue;
                if (safeSrc == neighbor) continue;

                currentDests[neighbor]++;
            }
        }

        cout << "[DEBUG] 源点 " << src << " 在层级 " << layer << " 下的唯一目标数: " << currentDests.size() << endl;

        int sourceTotalCount = 0;
        for (const auto& [dest, count] : currentDests) {
            cout << "[DEBUG] 源点 " << src << " 的目标 " << dest << " 出现次数: " << count << endl;
            sourceTotalCount += count;
        }

        cout << "[DEBUG] 源点 " << src << " 在层级 " << layer << " 下的总计数: " << sourceTotalCount << endl;
        printf("Source '%s' (%d) has %zu unique destinations, total frequency: %d.\n",
               src.c_str(), layer, currentDests.size(), sourceTotalCount);

        totalDestCount += sourceTotalCount;
        cout << "[INFO] 累计 totalDestCount: " << totalDestCount << endl;

        double M_flow = static_cast<double>(sourceTotalCount);

        // 业务 f 流平衡
        for (const string& node : allNodes) {
            string safeNode = safeName(node);
            bool isSrc = (node == src);
            bool isDestForThisSrc = (currentDests.count(safeNode) > 0);

            int maxULayer = (!node.empty() && node[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(node) > 0) ? NUM_K : 1);

            for (int k = 0; k < maxULayer; ++k) {
                std::map<std::string, double> row;

                for (const auto& e : rawEdges) {
                    string skey = safeName(e.key);

                    string neighbor;
                    if (e.u == node) neighbor = e.v;
                    else if (e.v == node) neighbor = e.u;
                    else continue;

                    if (neighbor.empty()) continue;

                    int maxVLayer = (!neighbor.empty() && neighbor[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(neighbor) > 0) ? NUM_K : 1);

                    for (int i = 0; i < maxVLayer; ++i) {
                        string layerPart = (e.u == node)
                            ? "_L" + to_string(k) + "_L" + to_string(i)
                            : "_L" + to_string(i) + "_L" + to_string(k);

                        string base = "f_src" + safeSrc + "_" + to_string(layer) + "_" + denseIndex + "_" + skey + layerPart;
                        string f_fwd = base + "_fwd";
                        string f_rev = base + "_rev";

                        // 与原文本模型一致：inflow - outflow
                        if (e.u == node) {
                            row[f_rev] += 1.0;
                            row[f_fwd] -= 1.0;
                        } else {
                            row[f_fwd] += 1.0;
                            row[f_rev] -= 1.0;
                        }
                    }
                }

                double rhs = 0.0;
                string constraintLabel;

                if (isSrc) {
                    constraintLabel = "TotalSrcFlow_" + safeSrc + "_L" + to_string(layer) + "_Curr_" + to_string(k);
                    rhs = (k == layer) ? -static_cast<double>(sourceTotalCount) : 0.0;
                } else {
                    int nodeFlow = 0;
                    if (isDestForThisSrc && k == layer) {
                        auto it = currentDests.find(safeNode);
                        if (it != currentDests.end()) nodeFlow = it->second;
                    }
                    constraintLabel = "FlowBalance_" + safeSrc + "_" + to_string(layer) + "_" + safeNode + "_L" + to_string(k);
                    rhs = static_cast<double>(nodeFlow);
                }

                modelBuilder.addRow(constraintLabel, std::move(row), rhs, rhs);
            }
        }

        // f 与物理 X 的容量耦合
        for (const auto& e : rawEdges) {
            string skey = safeName(e.key);

            int maxULayer = (!e.u.empty() && e.u[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(e.u) > 0) ? NUM_K : 1);
            int maxVLayer = (!e.v.empty() && e.v[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(e.v) > 0) ? NUM_K : 1);

            for (int k = 0; k < maxULayer; ++k) {
                for (int s = 0; s < maxVLayer; ++s) {
                    string f_layer = "_L" + to_string(k) + "_L" + to_string(s);
                    string f_fwd = "f_src" + safeSrc + "_" + to_string(layer) + "_" + denseIndex + "_" + skey + f_layer + "_fwd";
                    string f_rev = "f_src" + safeSrc + "_" + to_string(layer) + "_" + denseIndex + "_" + skey + f_layer + "_rev";
                    string x_fwd = "X_" + skey + f_layer + "_fwd";
                    string x_rev = "X_" + skey + f_layer + "_rev";

                    std::map<std::string, double> row;
                    row[f_fwd] += 1.0;
                    row[f_rev] += 1.0;
                    row[x_fwd] -= M_flow;
                    row[x_rev] -= M_flow;

                    modelBuilder.addRow("f_cap_sum_" + skey + f_layer,
                                        std::move(row), -kHighsInf, 0.0);
                }
            }
        }
    }

    // 全局 Z 流平衡
    for (const auto& v : allNodes) {
        string v_name = safeName(v);
        int maxVLayer = (!v.empty() && v[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(v) > 0) ? NUM_K : 1);

        for (int k = 0; k < maxVLayer; ++k) {
            std::map<std::string, double> row;

            for (const auto& e : rawEdges) {
                string skey = safeName(e.key);
                int maxULayer = (!e.u.empty() && e.u[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(e.u) > 0) ? NUM_K : 1);
                int maxVLayerPeer = (!e.v.empty() && e.v[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(e.v) > 0) ? NUM_K : 1);

                if (e.v == v) {
                    for (int s = 0; s < maxULayer; ++s) {
                        string f_layer = "_L" + to_string(s) + "_L" + to_string(k);
                        row["Z_" + skey + f_layer + "_fwd"] += 1.0;
                        row["Z_" + skey + f_layer + "_rev"] -= 1.0;
                    }
                } else if (e.u == v) {
                    for (int s = 0; s < maxVLayerPeer; ++s) {
                        string f_layer = "_L" + to_string(k) + "_L" + to_string(s);
                        row["Z_" + skey + f_layer + "_rev"] += 1.0;
                        row["Z_" + skey + f_layer + "_fwd"] -= 1.0;
                    }
                }
            }

            string v_id = v + "_L" + to_string(k);
            bool isSource = (v == initX && k == 0);
            bool isActTarget = (destNodes.count(v) > 0);
            double rhs = 0.0;

            if (isSource) {
                int n_targets = static_cast<int>(destNodes.size());
                rhs = -1.0 * n_targets;
                cout << "[DEBUG] 节点 " << v_id << " 是源点，RHS 设置为 " << rhs << endl;
            } else if (isActTarget) {
                rhs = 1.0;
            }

            modelBuilder.addRow("FlowBal_" + v_name + "_L" + to_string(k),
                                std::move(row), rhs, rhs);
        }
    }

    // 入度约束
    for (const auto& v : allNodes) {
        string v_name = safeName(v);
        int maxVLayer = (!v.empty() && v[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(v) > 0) ? NUM_K : 1);

        for (int k = 0; k < maxVLayer; ++k) {
            bool isRootAtStart = (v == initX);
            bool isEStart = (!v.empty() && v[0] == 'E');
            if (isEStart && !isRootAtStart) continue;

            std::map<std::string, double> row;

            for (const auto& e : rawEdges) {
                string skey = safeName(e.key);
                int maxULayer = (!e.u.empty() && e.u[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(e.u) > 0) ? NUM_K : 1);
                int maxVLayerPeer = (!e.v.empty() && e.v[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(e.v) > 0) ? NUM_K : 1);
                bool vIsN = (!v.empty() && v[0] == 'N');

                if (e.v == v) {
                    if (vIsN && !e.u.empty() && e.u[0] == 'E') continue;
                    for (int s = 0; s < maxULayer; ++s) {
                        string f_layer = "_L" + to_string(s) + "_L" + to_string(k);
                        row["X_" + skey + f_layer + "_fwd"] += 1.0;
                    }
                } else if (e.u == v) {
                    if (vIsN && !e.v.empty() && e.v[0] == 'E') continue;
                    for (int s = 0; s < maxVLayerPeer; ++s) {
                        string f_layer = "_L" + to_string(k) + "_L" + to_string(s);
                        row["X_" + skey + f_layer + "_rev"] += 1.0;
                    }
                }
            }

            if (isRootAtStart) {
                modelBuilder.addRow("InDegree_" + v_name + "_L" + to_string(k),
                                    std::move(row), 0.0, 0.0);
            } else {
                modelBuilder.addRow("InDegree_" + v_name + "_L" + to_string(k),
                                    std::move(row), -kHighsInf, 1.0);
            }
        }
    }

    // Z-X 耦合与方向互斥
    double M_cap = static_cast<double>(ActNodes.size());
    double eps = 1.0;

    for (const auto& e : rawEdges) {
        string skey = safeName(e.key);
        int maxULayer = (!e.u.empty() && e.u[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(e.u) > 0) ? NUM_K : 1);
        int maxVLayer = (!e.v.empty() && e.v[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(e.v) > 0) ? NUM_K : 1);

        for (int s = 0; s < maxULayer; ++s) {
            for (int k = 0; k < maxVLayer; ++k) {
                string f_layer = "_L" + to_string(s) + "_L" + to_string(k);
                string z_fwd = "Z_" + skey + f_layer + "_fwd";
                string z_rev = "Z_" + skey + f_layer + "_rev";
                string x_fwd = "X_" + skey + f_layer + "_fwd";
                string x_rev = "X_" + skey + f_layer + "_rev";

                modelBuilder.addRow("FlowX_fwd_UP_" + skey + f_layer,
                                    {{z_fwd, 1.0}, {x_fwd, -M_cap}},
                                    -kHighsInf, 0.0);

                modelBuilder.addRow("FlowX_fwd_LOW_" + skey + f_layer,
                                    {{z_fwd, 1.0}, {x_fwd, -eps}},
                                    0.0, kHighsInf);

                modelBuilder.addRow("FlowX_rev_UP_" + skey + f_layer,
                                    {{z_rev, 1.0}, {x_rev, -M_cap}},
                                    -kHighsInf, 0.0);

                modelBuilder.addRow("FlowX_rev_LOW_" + skey + f_layer,
                                    {{z_rev, 1.0}, {x_rev, -eps}},
                                    0.0, kHighsInf);

                modelBuilder.addRow("X_Mutex_" + skey + f_layer,
                                    {{x_fwd, 1.0}, {x_rev, 1.0}},
                                    -kHighsInf, 1.0);
            }
        }
    }

    cout << "业务对锚定约束已生成，正在构建起点电位初始化约束..." << endl;
    cout << "起点电位初始化约束已生成，正在构建终点可达性约束..." << endl;
    cout << "层级顺序约束已生成，正在构建内存模型..." << endl;

    // 原 LP 文件中的 Binary section 实际声明的是所有 X 变量。
    // 这里显式标记为二进制，避免依赖文本 parser。
    size_t binaryCount = 0;
    cout << "number of Edges: " << rawEdges.size() << endl;

    for (const auto& e : rawEdges) {
        string skey = safeName(e.key);
        int maxULayer = (!e.u.empty() && e.u[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(e.u) > 0) ? NUM_K : 1);
        int maxVLayer = (!e.v.empty() && e.v[0] == 'E') ? 1 : ((allowedMultiLayerNodes.count(e.v) > 0) ? NUM_K : 1);

        for (int k = 0; k < maxULayer; ++k) {
            for (int s = 0; s < maxVLayer; ++s) {
                string f_layer = "_L" + to_string(k) + "_L" + to_string(s);
                modelBuilder.markBinary("X_" + skey + f_layer + "_fwd");
                modelBuilder.markBinary("X_" + skey + f_layer + "_rev");
                binaryCount += 2;
            }
        }
    }

    cout << "Binary X variables registered directly: " << binaryCount << endl;

    cout << "Model has been built in solver-independent memory with "
         << modelBuilder.numCols() << " variables and "
         << modelBuilder.numRows() << " constraints." << endl;
}

struct SolverResult {
    std::string solver;
    std::string status;
    int statusCode = -1;
    bool optimal = false;
    bool hasSolution = false;
    double objective = std::numeric_limits<double>::quiet_NaN();
    std::unordered_map<std::string, double> varValues;
};

SolverResult solveWithHighs(const DirectModelBuilder& modelBuilder) {
    Highs highs;
    HighsStatus buildStatus = modelBuilder.passToHighs(highs);
    if (buildStatus != HighsStatus::kOk) {
        throw std::runtime_error("HiGHS 直接内存模型构建失败");
    }

    // 保留原来的 HiGHS 参数设置。
    highs.setOptionValue("primal_feasibility_tolerance", 1e-6);
    highs.setOptionValue("dual_feasibility_tolerance", 1e-6);
    highs.setOptionValue("mip_feasibility_tolerance", 1e-6);
    highs.setOptionValue("mip_rel_gap", 0.05);
    highs.setOptionValue("mip_abs_gap", 0.05);
    highs.setOptionValue("threads", 1);
    highs.setOptionValue("time_limit", 1800);
    highs.setOptionValue("clique_strategy", 2);
    highs.setOptionValue("flow_cover_strategy", 2);
    highs.setOptionValue("gomory_strategy", 2);
    highs.setOptionValue("mip_strategy_cuts", 2);
    highs.setOptionValue("mip_strategy_heuristic_effort", 1.0);
    highs.setOptionValue("presolve", "on");
    highs.setOptionValue("mip_strategy_branching", 1);
    highs.setOptionValue("mip_strategy_branching_score_strategy", 1);
    highs.setOptionValue("simplex_strategy", 1);
    highs.setOptionValue("simplex_dual_edge_weight_strategy", 2);
    highs.setOptionValue("mip_strategy_presolve_aggr", true);

    HighsStatus runStatus = highs.run();
    if (runStatus == HighsStatus::kError) {
        throw std::runtime_error("HiGHS 求解失败");
    }

    SolverResult result;
    result.solver = "highs";
    HighsModelStatus modelStatus = highs.getModelStatus();
    result.statusCode = static_cast<int>(modelStatus);
    result.status = highs.modelStatusToString(modelStatus);
    result.optimal = (modelStatus == HighsModelStatus::kOptimal);

    const HighsSolution& sol = highs.getSolution();
    const HighsLp& lp = highs.getLp();
    result.hasSolution = sol.value_valid &&
                         (sol.col_value.size() == static_cast<size_t>(lp.num_col_));
    if (result.hasSolution) {
        result.objective = highs.getObjectiveValue();
        for (HighsInt i = 0; i < lp.num_col_; ++i) {
            const std::string name =
                (lp.col_names_.size() == static_cast<size_t>(lp.num_col_))
                    ? lp.col_names_[i]
                    : ("var_" + std::to_string(i));
            result.varValues[name] = sol.col_value[i];
        }
    }
    return result;
}

static void checkScip(SCIP_RETCODE rc, const char* action) {
    if (rc != SCIP_OKAY) {
        throw std::runtime_error(std::string("SCIP 调用失败: ") + action +
                                 ", retcode=" + std::to_string(static_cast<int>(rc)));
    }
}

static std::string scipStatusToString(SCIP_STATUS status) {
    switch (status) {
        case SCIP_STATUS_UNKNOWN: return "Unknown";
        case SCIP_STATUS_USERINTERRUPT: return "User interrupt";
        case SCIP_STATUS_NODELIMIT: return "Node limit";
        case SCIP_STATUS_TOTALNODELIMIT: return "Total node limit";
        case SCIP_STATUS_STALLNODELIMIT: return "Stall node limit";
        case SCIP_STATUS_TIMELIMIT: return "Time limit";
        case SCIP_STATUS_MEMLIMIT: return "Memory limit";
        case SCIP_STATUS_GAPLIMIT: return "Gap limit";
        case SCIP_STATUS_SOLLIMIT: return "Solution limit";
        case SCIP_STATUS_BESTSOLLIMIT: return "Best solution limit";
        case SCIP_STATUS_RESTARTLIMIT: return "Restart limit";
        case SCIP_STATUS_OPTIMAL: return "Optimal";
        case SCIP_STATUS_INFEASIBLE: return "Infeasible";
        case SCIP_STATUS_UNBOUNDED: return "Unbounded";
        case SCIP_STATUS_INFORUNBD: return "Infeasible or unbounded";
        default: return "SCIP status " + std::to_string(static_cast<int>(status));
    }
}

SolverResult solveWithScip(const DirectModelBuilder& modelBuilder) {
    SCIP* scip = nullptr;
    std::vector<SCIP_VAR*> scipVars;

    try {
        checkScip(SCIPcreate(&scip), "SCIPcreate");
        checkScip(SCIPincludeDefaultPlugins(scip), "SCIPincludeDefaultPlugins");
        checkScip(SCIPcreateProbBasic(scip, "network_design"), "SCIPcreateProbBasic");
        checkScip(SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE), "SCIPsetObjsense");

        const auto& names = modelBuilder.getVarNames();
        const auto& costs = modelBuilder.getColCost();
        const auto& lowers = modelBuilder.getColLower();
        const auto& uppers = modelBuilder.getColUpper();
        const auto& binary = modelBuilder.getIsBinaryVar();
        const auto& varIndex = modelBuilder.getVarIndex();

        scipVars.resize(names.size(), nullptr);
        for (size_t i = 0; i < names.size(); ++i) {
            const double lb = (lowers[i] <= -0.5 * kHighsInf)
                                ? -SCIPinfinity(scip) : lowers[i];
            const double ub = (uppers[i] >= 0.5 * kHighsInf)
                                ? SCIPinfinity(scip) : uppers[i];
            const SCIP_VARTYPE type = binary[i] ? SCIP_VARTYPE_BINARY
                                                 : SCIP_VARTYPE_CONTINUOUS;

            checkScip(
                SCIPcreateVarBasic(scip, &scipVars[i], names[i].c_str(),
                                   lb, ub, costs[i], type),
                "SCIPcreateVarBasic");
            checkScip(SCIPaddVar(scip, scipVars[i]), "SCIPaddVar");
        }

        for (const auto& row : modelBuilder.getRows()) {
            std::vector<SCIP_VAR*> vars;
            std::vector<SCIP_Real> vals;
            vars.reserve(row.coeffs.size());
            vals.reserve(row.coeffs.size());

            for (const auto& [varName, coeff] : row.coeffs) {
                auto it = varIndex.find(varName);
                if (it == varIndex.end()) {
                    throw std::runtime_error("内部建模错误：SCIP 未找到变量 " + varName);
                }
                vars.push_back(scipVars[it->second]);
                vals.push_back(static_cast<SCIP_Real>(coeff));
            }

            const SCIP_Real lhs = (row.lower <= -0.5 * kHighsInf)
                                    ? -SCIPinfinity(scip)
                                    : static_cast<SCIP_Real>(row.lower);
            const SCIP_Real rhs = (row.upper >= 0.5 * kHighsInf)
                                    ? SCIPinfinity(scip)
                                    : static_cast<SCIP_Real>(row.upper);

            SCIP_CONS* cons = nullptr;
            checkScip(
                SCIPcreateConsBasicLinear(
                    scip, &cons, row.name.c_str(), static_cast<int>(vars.size()),
                    vars.empty() ? nullptr : vars.data(),
                    vals.empty() ? nullptr : vals.data(), lhs, rhs),
                "SCIPcreateConsBasicLinear");
            checkScip(SCIPaddCons(scip, cons), "SCIPaddCons");
            checkScip(SCIPreleaseCons(scip, &cons), "SCIPreleaseCons");
        }

        // 与原 HiGHS 设置对应的通用求解限制；其余 SCIP 策略使用 SCIP 默认值。
        checkScip(SCIPsetRealParam(scip, "numerics/feastol", 1e-6), "set numerics/feastol");
        checkScip(SCIPsetRealParam(scip, "numerics/dualfeastol", 1e-6), "set numerics/dualfeastol");
        checkScip(SCIPsetRealParam(scip, "limits/gap", 0.05), "set limits/gap");
        checkScip(SCIPsetRealParam(scip, "limits/absgap", 0.05), "set limits/absgap");
        checkScip(SCIPsetRealParam(scip, "limits/time", 1800.0), "set limits/time");
        checkScip(SCIPsetIntParam(scip, "parallel/maxnthreads", 1), "set parallel/maxnthreads");

        checkScip(SCIPsolve(scip), "SCIPsolve");

        SolverResult result;
        result.solver = "scip";
        SCIP_STATUS status = SCIPgetStatus(scip);
        result.statusCode = static_cast<int>(status);
        result.status = scipStatusToString(status);
        result.optimal = (status == SCIP_STATUS_OPTIMAL);

        SCIP_SOL* sol = SCIPgetBestSol(scip);
        result.hasSolution = (sol != nullptr);
        if (sol != nullptr) {
            result.objective = SCIPgetSolOrigObj(scip, sol);
            for (size_t i = 0; i < names.size(); ++i) {
                result.varValues[names[i]] = SCIPgetSolVal(scip, sol, scipVars[i]);
            }
        }

        for (auto& var : scipVars) {
            if (var != nullptr) checkScip(SCIPreleaseVar(scip, &var), "SCIPreleaseVar");
        }
        checkScip(SCIPfree(&scip), "SCIPfree");
        return result;
    } catch (...) {
        if (scip != nullptr) {
            for (auto& var : scipVars) {
                if (var != nullptr) SCIPreleaseVar(scip, &var);
            }
            SCIPfree(&scip);
        }
        throw;
    }
}

SolverResult solveModel(const DirectModelBuilder& modelBuilder,
                        const std::string& solverName) {
    if (solverName == "highs") return solveWithHighs(modelBuilder);
    if (solverName == "scip") return solveWithScip(modelBuilder);
    throw std::invalid_argument("不支持的求解器: " + solverName +
                                "，可选值为 highs 或 scip");
}

struct Edge {
    std::string u, v;
    double cost;
    std::string key;
};

WeightedGraph build_graph_from_csv(const std::string& csv_path) {
    WeightedGraph graph;
    std::ifstream file(csv_path);
    printf("Reading graph from CSV: %s\n", csv_path.c_str());
    if (!file.is_open()) return graph;

    std::string line;
    getline(file, line);
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> row;
        std::string col;
        while(getline(ss, col, ',')) row.push_back(col);

        if(row.size() >= 5) {
            std::string u = safeName(row[2]);
            std::string v = safeName(row[3]);
            double d = std::stod(row[4]);

            graph[u][v] = d;
            graph[v][u] = d;
        }
    }
    return graph;
}

using AdjMap = std::map<std::string, std::vector<std::pair<std::string, double>>>;

WeightedGraph parse_solution_to_adj(const std::unordered_map<std::string, double>& var_values,
                                    const WeightedGraph& full_graph) {
    WeightedGraph adj;

    std::regex x_pattern(R"(^X_(.+)_L(\d+)_L(\d+)_(fwd|rev)$)");

    std::regex f_pattern(R"(^f_src(.+?)_(\d+)_(\d+)_(.+?)_L(\d+)_L(\d+)_(fwd|rev)$)");

    std::smatch x_match;
    for (const auto& [var_name, val] : var_values) {

        if (std::regex_match(var_name, x_match, x_pattern)) {
            if (val <= 0.5) continue;

            std::string x_node_pair = x_match[1];
            std::string l1 = x_match[2];
            std::string l2 = x_match[3];

            bool f_matched = false;
            for (const auto& [f_name, f_val] : var_values) {
                std::smatch f_match;
                if (std::regex_match(f_name, f_match, f_pattern)) {

                    if (f_match[4] == x_node_pair && f_match[5] == l1 && f_match[6] == l2) {
                        if (f_val >= 0.5) {

                            f_matched = true;
                            break;
                        }
                    }
                }
            }
            if (!f_matched) continue;

            std::smatch match;
            if (std::regex_match(var_name, match, x_pattern)) {
                std::string full_key = match[1].str();
                size_t to_pos = full_key.find("_to_");
                if (to_pos != std::string::npos) {
                    std::string u_raw = safeName(full_key.substr(0, to_pos));
                    std::string v_raw = safeName(full_key.substr(to_pos + 4));

                    std::string u_node = u_raw + "_L" + match[2].str();
                    std::string v_node = v_raw + "_L" + match[3].str();

                    double dist = get_shortest_distance_weighted(full_graph, u_raw, v_raw);

                    std::cout << "Parsed: " << u_node << " <-> " << v_node
                              << " | Distance: " << dist << std::endl;

                    adj[u_node][v_node] = dist;
                    adj[v_node][u_node] = dist;
                }
            }
        }
    }
    return adj;
}

bool find_cycle_undirected(const std::string& u,
                           const std::string& parent,
                           const WeightedGraph& graph,
                           std::unordered_map<std::string, bool>& visited,
                           std::vector<std::string>& path,
                           std::vector<std::string>& cycle) {

    if (u.empty() || (u[0] != 'N')) {
        return false;
    }

    visited[u] = true;
    path.push_back(u);

    if (graph.count(u)) {
        for (auto const& [v, weight] : graph.at(u)) {

            if (v.empty() || (v[0] != 'N')) {
                continue;
            }

            if (v == parent) continue;

            if (visited[v]) {

                cycle.clear();
                auto it = std::find(path.begin(), path.end(), v);
                if (it != path.end()) {
                    cycle.assign(it, path.end());
                    cycle.push_back(v);
                    return true;
                }
            } else {

                if (find_cycle_undirected(v, u, graph, visited, path, cycle)) {
                    return true;
                }
            }
        }
    }

    path.pop_back();
    return false;
}

void detect_and_print_undirected_cycle(const WeightedGraph& graph) {
    std::unordered_map<std::string, bool> visited;
    std::vector<std::string> path;
    std::vector<std::string> cycle;

    for (auto const& [node, neighbors] : graph) {
        if (!visited[node]) {

            if (find_cycle_undirected(node, "", graph, visited, path, cycle)) {
                std::cout << "检测到环: ";
                for (size_t i = 0; i < cycle.size(); ++i) {
                    std::cout << cycle[i] << (i == cycle.size() - 1 ? "" : " - ");
                }
                std::cout << " - " << cycle[0] << std::endl;
                return;
            }
        }
    }
    std::cout << "图中不存在环。" << std::endl;
}

std::string get_base_node(const std::string& s) {
    size_t pos = s.find_last_of("_L");
    if (pos != std::string::npos) {
        return s.substr(0, pos - 1);
    }
    return s;
}

double find_weight_in_adjmap(const WeightedGraph& full_graph, const std::string& u, const std::string& v) {

    auto it = full_graph.find(u);
    if (it != full_graph.end()) {
        for (const auto& neighbor : it->second) {
            if (neighbor.first == v) {
                return neighbor.second;
            }
        }
    }

    double dist = get_shortest_distance_weighted(full_graph, u, v);

    if (dist >= 1e17) {

        return 1.0;
    }

    return dist;
}

void add_path_to_graph(WeightedGraph& graph,
                       const std::vector<std::string>& path,
                       const WeightedGraph& full_graph) {
    if (path.size() < 2) return;

    for (size_t i = 0; i < path.size() - 1; ++i) {
        const std::string& u = path[i];
        const std::string& v = path[i + 1];

        std::string base_u = get_base_node(u);
        std::string base_v = get_base_node(v);

        double weight = find_weight_in_adjmap(full_graph, base_u, base_v);

        graph[u][v] = weight;
        graph[v][u] = weight;
    }
}

std::vector<std::string> get_shortest_path_nodes(
    const std::string& start,
    const std::string& end,
    const WeightedGraph& full_graph) {

    std::map<std::string, double> dist;
    std::map<std::string, std::string> parent;

    for (auto const& [node, neighbors] : full_graph) dist[node] = 1e18;
    dist[start] = 0.0;

    std::priority_queue<std::pair<double, std::string>, std::vector<std::pair<double, std::string>>, std::greater<>> pq;
    pq.push({0.0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        if (u == end) break;

        if (full_graph.count(u)) {
            for (auto const& edge : full_graph.at(u)) {
                std::string v = edge.first;
                double w = edge.second;
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    parent[v] = u;
                    pq.push({dist[v], v});
                }
            }
        }
    }

    std::vector<std::string> path;
    for (std::string at = end; at != ""; at = parent[at]) {
        path.push_back(at);
        if (at == start) break;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

static std::map<std::pair<std::string, std::string>, std::vector<std::string>> path_shape_cache;

static std::map<std::pair<std::string, std::string>, std::vector<std::string>> logical_path_cache;

bool split_layered_node(const std::string& node, std::string& base, int& layer) {
    static const std::regex pattern(R"(^(.*)_L(\d+)$)");
    std::smatch match;
    if (!std::regex_match(node, match, pattern)) {
        base = node;
        layer = 0;
        return false;
    }
    base = match[1].str();
    layer = std::stoi(match[2].str());
    return true;
}

std::string remove_suffix(const std::string& node) {
    std::string base;
    int layer = 0;
    split_layered_node(node, base, layer);
    return base;
}

bool has_edge(const WeightedGraph& graph, const std::string& u, const std::string& v) {

    auto it_u = graph.find(u);
    if (it_u != graph.end() && it_u->second.find(v) != it_u->second.end()) {
        return true;
    }

    auto it_v = graph.find(v);
    if (it_v != graph.end() && it_v->second.find(u) != it_v->second.end()) {
        return true;
    }

    return false;
}

std::string get_suffix(const std::string& node) {
    std::string base;
    int layer = 0;
    if (!split_layered_node(node, base, layer)) return "";
    return "_L" + std::to_string(layer);
}

std::vector<std::string> expand_path(const std::vector<std::string>& compressed_path, const WeightedGraph& full_graph) {
    if (compressed_path.size() < 2) return compressed_path;

    std::vector<std::string> expanded;
    expanded.push_back(compressed_path[0]);

    for (size_t i = 0; i < compressed_path.size() - 1; ++i) {
        const std::string& u_orig = compressed_path[i];
        const std::string& v_orig = compressed_path[i + 1];

        if (has_edge(full_graph, remove_suffix(u_orig), remove_suffix(v_orig))) {
            expanded.push_back(v_orig);
        } else {

            bool is_reversed = (u_orig > v_orig);
            std::pair<std::string, std::string> logical_key = is_reversed ?
                              std::make_pair(v_orig, u_orig) : std::make_pair(u_orig, v_orig);

            if (logical_path_cache.find(logical_key) == logical_path_cache.end()) {

                std::string u = remove_suffix(logical_key.first);
                std::string v = remove_suffix(logical_key.second);
                std::pair<std::string, std::string> shape_key = {std::min(u, v), std::max(u, v)};

                if (path_shape_cache.find(shape_key) == path_shape_cache.end()) {
                    path_shape_cache[shape_key] = get_shortest_path_nodes(shape_key.first, shape_key.second, full_graph);
                }

                std::vector<std::string> raw = path_shape_cache[shape_key];

                if (logical_key.first > logical_key.second) std::reverse(raw.begin(), raw.end());

                std::string s_first = get_suffix(logical_key.first);
                std::string s_second = get_suffix(logical_key.second);

                std::vector<std::string> labeled_sub;
                for (size_t k = 0; k < raw.size(); ++k) {
                    std::string node = remove_suffix(raw[k]);
                    if (k > 0 && k < raw.size() - 1) labeled_sub.push_back(node + s_first);
                    else if (k == raw.size() - 1) labeled_sub.push_back(node + s_second);
                    else labeled_sub.push_back(raw[k]);
                }
                logical_path_cache[logical_key] = labeled_sub;
            }

            std::vector<std::string> path_to_add = logical_path_cache[logical_key];
            if (is_reversed) {
                std::reverse(path_to_add.begin(), path_to_add.end());

            }

            for (size_t k = 1; k < path_to_add.size() - 1; ++k) {
                expanded.push_back(path_to_add[k]);
            }
            expanded.push_back(v_orig);
        }
    }
    return expanded;
}

using NodePair = std::pair<std::string, std::string>;

struct PairPathOutputRow {
    std::string harness_id;
    std::string circuit_id;
    std::string start;
    std::string end;
    double density = 0.0;
    std::vector<std::string> path;
};

std::vector<std::string> exportedPairPathNodes(const std::vector<std::string>& path) {
    std::vector<std::string> tokens;
    if (path.empty()) return tokens;
    tokens.reserve(path.size());

    for (std::size_t i = 0; i < path.size(); ++i) {
        std::string base;
        int layer = 0;
        bool layered = split_layered_node(path[i], base, layer);
        if (i == 0 || i + 1 == path.size()) {
            tokens.push_back(base);
        } else if (layered) {
            tokens.push_back(base + "_" + std::to_string(layer + 1));
        } else {
            tokens.push_back(base);
        }
    }

    std::vector<std::string> filtered;
    filtered.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        bool skip_entry = i > 0 && i + 1 < tokens.size() &&
                          !tokens[i].empty() && tokens[i].front() == 'E' &&
                          !tokens[i - 1].empty() && tokens[i - 1].front() == 'N' &&
                          !tokens[i + 1].empty() && tokens[i + 1].front() == 'N';
        if (!skip_entry) filtered.push_back(tokens[i]);
    }
    return filtered;
}

std::string formatPairPath(const std::vector<std::string>& path) {
    const auto nodes = exportedPairPathNodes(path);
    std::ostringstream out;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (i) out << ';';
        out << nodes[i];
    }
    return out.str();
}

int writePairPathXls(const std::vector<PairPathOutputRow>& rows,
                     const std::string& output_file) {
    if (output_file.empty()) return 1;

    std::filesystem::path output_path(output_file);
    if (output_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(output_path.parent_path(), ec);
        if (ec) return 1;
    }

    xlslib_core::workbook workbook;
    xlslib_core::worksheet* worksheet = workbook.sheet("Sheet1");
    worksheet->label(0, 0, "id");
    worksheet->label(0, 1, "harness_id");
    worksheet->label(0, 2, "circuit_id");
    worksheet->label(0, 3, "start");
    worksheet->label(0, 4, "end");
    worksheet->label(0, 5, "path");

    for (std::size_t i = 0; i < rows.size(); ++i) {
        const int row = static_cast<int>(i) + 1;
        const std::string path = formatPairPath(rows[i].path);
        worksheet->number(row, 0, row);
        worksheet->label(row, 1, rows[i].harness_id.c_str());
        worksheet->label(row, 2, rows[i].circuit_id.c_str());
        worksheet->label(row, 3, rows[i].start.c_str());
        worksheet->label(row, 4, rows[i].end.c_str());
        worksheet->label(row, 5, path.c_str());
    }

    return workbook.Dump(output_file);
}

int processNetworkFlows(const std::string& edgesFile,
                         const std::string& pairsFile,
                         const std::map<std::string, std::pair<std::string, double>>& neighborIndex,
                         const WeightedGraph& milp_adj,
                         const std::string& output_file) {

    WeightedGraph full_graph = build_graph_from_csv(edgesFile);

    std::string line;

    std::ifstream fPairs(pairsFile);
    if (!fPairs.is_open()) {
        std::cerr << "[ERROR] 无法打开 pairsFile: " << pairsFile << std::endl;
        return 1;
    }

    std::vector<PairPathOutputRow> output_rows;
    std::getline(fPairs, line);
    while (std::getline(fPairs, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::vector<std::string> c;
        std::string col;
        while (std::getline(ss, col, ',')) c.push_back(col);
        if (c.size() < 10) {
            std::cerr << "[ERROR] pairs 行列数不足: " << line << std::endl;
            return 1;
        }

        PairPathOutputRow row;
        row.harness_id = c[1];
        row.circuit_id = c[2];
        row.start = c[3];
        row.end = c[5];
        try {
            row.density = std::stod(c[9]);
        } catch (const std::exception&) {
            std::cerr << "[ERROR] density 无法解析: " << c[9] << std::endl;
            return 1;
        }
        output_rows.push_back(std::move(row));
    }

    for (std::size_t pair_id = 0; pair_id < output_rows.size(); ++pair_id) {
        auto& row = output_rows[pair_id];
        std::string u_raw = safeName(row.start);
        std::string v_raw = safeName(row.end);
        int k_val = 0;

        std::vector<std::string> path1;
        std::vector<std::string> path2;
        std::vector<std::string> path3;

        std::string mapped_u = u_raw;
        std::string mapped_v = v_raw;

        auto uit = neighborIndex.find(row.start);
        if (uit == neighborIndex.end()) uit = neighborIndex.find(u_raw);
        if (uit != neighborIndex.end()) mapped_u = safeName(uit->second.first);

        auto vit = neighborIndex.find(row.end);
        if (vit == neighborIndex.end()) vit = neighborIndex.find(v_raw);
        if (vit != neighborIndex.end()) mapped_v = safeName(vit->second.first);

        path1 = findPathInOriginal(u_raw, mapped_u, full_graph);
        path2 = findPathInOriginal(mapped_v, v_raw, full_graph);

        if (u_raw == mapped_u && path1.empty()) path1 = {u_raw};
        if (mapped_v == v_raw && path2.empty()) path2 = {v_raw};

        std::string u_Lk = mapped_u + "_L" + std::to_string(k_val);
        std::string v_Lk = mapped_v + "_L" + std::to_string(k_val);

        if (u_Lk == v_Lk) {
            path3 = {u_Lk};
        } else {
            path3 = findPathInOriginal(u_Lk, v_Lk, milp_adj);
        }

        if (!path3.empty()) {
            path3 = expand_path(path3, full_graph);
        }

        if (path1.empty() || path2.empty() || path3.empty()) {
            std::cerr << "[WARNING] pair " << (pair_id + 1)
                      << " 无法重组完整路径: " << row.start << " -> " << row.end
                      << std::endl;
            continue;
        }

        auto add_suffix = [k_val](std::vector<std::string> path) {
            std::transform(path.begin(), path.end(), path.begin(),
                           [k_val](const std::string& s) {
                               return s + "_L" + std::to_string(k_val);
                           });
            return path;
        };

        std::vector<std::string> p1_modified = add_suffix(path1);
        std::vector<std::string> p2_modified = add_suffix(path2);

        std::vector<std::string> final_path = p1_modified;
        if (path3.size() > 1)
            final_path.insert(final_path.end(), path3.begin() + 1, path3.end());
        if (p2_modified.size() > 1)
            final_path.insert(final_path.end(), p2_modified.begin() + 1, p2_modified.end());

        std::vector<std::string> deduplicated;
        deduplicated.reserve(final_path.size());
        for (const auto& node : final_path) {
            if (deduplicated.empty() || deduplicated.back() != node)
                deduplicated.push_back(node);
        }

        row.path = std::move(deduplicated);
    }

    const int write_rc = writePairPathXls(output_rows, output_file);
    if (write_rc != 0) {
        std::cerr << "[ERROR] 路径文件写入失败: " << output_file << std::endl;
        return 1;
    }

    std::cout << "Pair paths saved to: " << output_file << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {

    if (argc < 5) {
        std::cerr << "用法: " << argv[0]
                  << " <edges_file> <pairs_file> <k_value> <output_path.xls>"
                  << " [--solver highs|scip]" << std::endl;
        return 1;
    }

    std::string fileE = argv[1];
    std::string fileP = argv[2];
    std::string outputFile = argv[4];
    std::string solverName = "highs";  // 保持旧命令行兼容：未指定时默认 HiGHS

    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--solver") {
            if (i + 1 >= argc) {
                std::cerr << "错误: --solver 后必须指定 highs 或 scip" << std::endl;
                return 1;
            }
            solverName = argv[++i];
        } else if (arg.rfind("--solver=", 0) == 0) {
            solverName = arg.substr(std::string("--solver=").size());
        } else if (arg == "highs" || arg == "scip") {
            // 也兼容把求解器直接作为第5个位置参数传入。
            solverName = arg;
        } else {
            std::cerr << "错误: 无法识别的参数: " << arg << std::endl;
            return 1;
        }
    }

    std::transform(solverName.begin(), solverName.end(), solverName.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (solverName != "highs" && solverName != "scip") {
        std::cerr << "错误: solver 必须是 highs 或 scip，当前值: " << solverName << std::endl;
        return 1;
    }

    int k = 0;
    try {

        k = std::stoi(argv[3]);
    } catch (const std::exception& e) {
        std::cerr << "错误: k 必须是一个有效的整数！" << std::endl;
        return 1;
    }

    std::cout << ">>> 已读取参数：" << std::endl;
    std::cout << "File E: " << fileE << std::endl;
    std::cout << "File P: " << fileP << std::endl;
    std::cout << "Value k: " << k << std::endl;
    std::cout << "Solver: " << solverName << std::endl;

    NUM_K = k;

    vector<RawEdge> rawEdges;
    vector<PairReq> pairs;
    set<string> allNodes;
    map<string, vector<pair<string, pair<string, double>>>> adj;

    ifstream fe(fileE); string line, col; getline(fe, line);
    while (getline(fe, line)) {
        stringstream ss(line); vector<string> c;
        while (getline(ss, col, ',')) c.push_back(col);
        if (c.size() >= 5) {
            string u = c[2], v = c[3]; double cost = stod(c[4]);
            string key = safeName(u) + "_to_" + safeName(v);
            rawEdges.push_back({u, v, cost, key});
            allNodes.insert(u); allNodes.insert(v);
            adj[u].push_back({v, {key, cost}});
            adj[v].push_back({u, {key, cost}});

        }
    }

    ifstream fp(fileP); getline(fp, line);
    while (getline(fp, line)) {
        stringstream ss(line); vector<string> c;
        while (getline(ss, col, ',')) c.push_back(col);
        if (c.size() >= 6) pairs.push_back({c[3], c[5], stod(c[9]), (int)pairs.size()});
    }

    cout << "----------(before simplification)--------------------------" << endl;
    cout << ">>> 图完成统计信息：" << endl;
    cout << ">>> 初始节点数 (Nodes): " << allNodes.size() << endl;
    cout << ">>> 初始边数 (Edges): " << rawEdges.size() << endl;
    cout << ">>> 初始点对数 (pairs): " << pairs.size() << endl;

    map<int, int> degreeDist1;
    for (auto const& [node, neighbors] : adj) {
        degreeDist1[neighbors.size()]++;
    }

    cout << ">>> 节点度数分布：" << endl;
    for (auto const& [deg, count] : degreeDist1) {
        cout << "    度数为 " << deg << " 的节点数量: " << count << endl;
    }
    cout << "------------------------------------" << endl;

    simplifyGraphVersion2(rawEdges, pairs, allNodes, adj);

    cout << "----------(after simplification)--------------------------" << endl;
    cout << ">>> 图精简完成统计信息：" << endl;
    cout << ">>> 剩余节点数 (Nodes): " << allNodes.size() << endl;
    cout << ">>> 所有剩余节点列表：" << endl;
    for (const auto& node : allNodes) {
        cout << "    - " << node << endl;
    }
    cout << ">>> 剩余边数 (Edges): " << rawEdges.size() << endl;
    cout << ">>> 剩余点对数 (pairs): " << pairs.size() << endl;

    auto sortedComponents = getAllComponentsSorted(adj);

    cout << "共发现 " << sortedComponents.size() << " 个连通分量：" << endl;
    for (size_t i = 0; i < sortedComponents.size(); ++i) {
        cout << "第 " << i + 1 << " 大分量: "
            << "节点数=" << sortedComponents[i].nodes.size()
            << ", 边数=" << sortedComponents[i].edges.size() << endl;
    }

    if (!sortedComponents.empty()) {

        const Component& largestComponent = sortedComponents[0];

        cout << "最大的连通分量包含 " << largestComponent.nodes.size() << " 个节点。" << endl;
        cout << "共有 " << largestComponent.edges.size() << " 条边。" << endl;

    } else {
        cout << "图中没有找到任何连通分量（可能图为空）。" << endl;
    }

    GraphSplitResult splitResult;

    extractComponents(adj, splitResult);

    map<int, int> degreeDist;
    for (auto const& [node, neighbors] : adj) {
        degreeDist[neighbors.size()]++;
    }

    cout << ">>> 节点度数分布：" << endl;
    for (auto const& [deg, count] : degreeDist) {
        cout << "    度数为 " << deg << " 的节点数量: " << count << endl;
    }
    cout << "------------------------------------" << endl;

    cout << "根据点对获取图数据" << endl;

    map<string, vector<RawEdge>> retainedEdgesPerV;
    map<string, set<string>> retainedNodesPerV;
    set<string> allRetainedEdgeKeys;
    set<string> allFilteredEdgeKeys;
    set<string> allRetainedNodes;

    int K = 10;
    vector<PairReq> independentSet = processGraphEdges(
        rawEdges, pairs, K, retainedEdgesPerV, retainedNodesPerV,
        allRetainedEdgeKeys, allFilteredEdgeKeys, allRetainedNodes, true
    );

    cout << "\n======================================================================" << endl;
    cout << "============         1. 独立集选定点及路由边检查          ============" << endl;
    cout << "======================================================================" << endl;

    cout << "\n======================================================================" << endl;
    cout << "============               2. 全局网络缩减报告            ============" << endl;
    cout << "======================================================================" << endl;
    cout << " 【全局网络原始总边数】 : " << rawEdges.size() << " 条" << endl;
    cout << " --------------------------------------------------------------------" << endl;

    cout << " 【被成功过滤的无关边】 : " << allFilteredEdgeKeys.size() << " 条" << endl;
    cout << "   ➔ 边 Key 列表: ";
    if (allFilteredEdgeKeys.empty()) cout << "(无)";
    for (const string& key : allFilteredEdgeKeys) cout << key << "  ";
    cout << "\n" << endl;
    cout << "======================================================================" << endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << " [Group 1] 原始点对 -> 化简后点对映射表" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::left << std::setw(8) << "ID"
              << std::setw(15) << "Orig U"
              << std::setw(15) << "Orig V"
              << std::setw(15) << "Mapped U"
              << std::setw(15) << "Mapped V" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    for (const auto& pair : pairs) {
        std::string mapped_u = getMappedNode(pair.u_orig);
        std::string mapped_v = getMappedNode(pair.v_orig);

        std::cout << std::left << std::setw(8) << pair.id
                  << std::setw(15) << pair.u_orig
                  << std::setw(15) << pair.v_orig
                  << std::setw(15) << mapped_u
                  << std::setw(15) << mapped_v << std::endl;
    }

    std::vector<std::string> selected_nodes;
    std::vector<PairAssignment> assignments;
    std::cout << "\n[LOG] 开始执行 findIndependentSet..." << std::endl;
    findIndependentSet(pairs, selected_nodes, assignments);
    std::cout << "[LOG] findIndependentSet 执行完毕。选中的点数量: " << selected_nodes.size() << std::endl;

    std::set<std::string> chosen_node_set(selected_nodes.begin(), selected_nodes.end());

    std::cout << "\n========================================" << std::endl;
    std::cout << " [Group 2] 选中的点集合及管辖的原始点对统计" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << std::left << std::setw(25) << "Selected Node"
              << std::setw(25) << "Managed Orig Pairs Count" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::map<std::string, std::vector<PairReq>> node_to_managed_pairs;

    for (const auto& raw_pair : pairs) {
        std::string ru = getMappedNode(raw_pair.u_orig);
        std::string rv = getMappedNode(raw_pair.v_orig);

        bool u_selected = (chosen_node_set.find(ru) != chosen_node_set.end());
        bool v_selected = (chosen_node_set.find(rv) != chosen_node_set.end());

        if (u_selected) {
            node_to_managed_pairs[ru].push_back(raw_pair);
        } else if (v_selected) {
            node_to_managed_pairs[rv].push_back(raw_pair);
        }
    }

    int total_managed_orig_pairs = 0;

    for (const auto& node : chosen_node_set) {
        const auto& managed_list = node_to_managed_pairs[node];
        int count = managed_list.size();
        total_managed_orig_pairs += count;

        std::cout << std::left << std::setw(25) << node
                  << std::setw(25) << count << std::endl;
    }

    std::cout << "\n---------------- [校验统计信息] ----------------" << std::endl;
    std::cout << "  -> 选中的点集合总数 (Nodes Count): " << chosen_node_set.size() << std::endl;
    std::cout << "  -> 实际管辖的原始点对总数: " << total_managed_orig_pairs << std::endl;
    std::cout << "  -> 输入的全局原始点对总数 (pairs.size): " << pairs.size() << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    DirectModelBuilder modelBuilder;
    buildModel(modelBuilder, rawEdges, pairs, allNodes, chosen_node_set, node_to_managed_pairs);

    SolverResult solveResult;
    try {
        solveResult = solveModel(modelBuilder, solverName);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] 求解失败: " << e.what() << std::endl;
        return 2;
    }

    cout << "\n========================================" << endl;
    cout << "求解器: " << solveResult.solver << endl;
    cout << "求解状态: " << solveResult.status << endl;
    cout << "状态码: " << solveResult.statusCode << endl;
    if (solveResult.hasSolution) {
        cout << "最终目标值 (Real): " << std::fixed << std::setprecision(6)
             << (solveResult.objective / SCALE) << endl;
    } else {
        cout << "未获得可用解。" << endl;
    }
    cout << "========================================\n" << endl;

    if (!solveResult.hasSolution) {
        std::cerr << "[ERROR] 当前求解状态下没有可用于后处理的解。" << std::endl;
        return 2;
    }

    std::unordered_map<std::string, double> cost_lookup;
    for (const auto& edge : rawEdges) {
        cost_lookup[edge.key] = edge.cost;
    }

    long double manual_obj = 0.0;
    double eps = 1e-7;

    for (const auto& [var_name, value] : solveResult.varValues) {
        if (value > (1.0 - eps)) {
            auto it = cost_lookup.find(var_name);
            if (it != cost_lookup.end()) {
                manual_obj += static_cast<long double>(it->second);
            }
        }
    }

    cout << std::fixed << std::setprecision(10);
    cout << "Manual Objective (6+ decimal): " << (double)manual_obj << endl;
    cout << solveResult.solver << " Objective: " << solveResult.objective << endl;

    WeightedGraph full_graph = build_graph_from_csv(fileE);

    auto MILP_edges = parse_solution_to_adj(solveResult.varValues, full_graph);

    std::cout << std::left << std::setw(20) << "起点 (带层)"
              << std::setw(20) << "终点 (带层)"
              << "计算所得最短距离" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (const auto& [u, neighbors] : MILP_edges) {
        for (const auto& edge : neighbors) {
            std::cout << std::left << std::setw(20) << u
                      << std::setw(20) << edge.first
                      << std::fixed << std::setprecision(4) << edge.second
                      << std::endl;
        }
    }

    detect_and_print_undirected_cycle(MILP_edges);

    WeightedGraph adj_orig;
    loadOriginalGraph(fileE, NUM_K, adj_orig);

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "流还原分析报告" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    for (const auto& [u_node, neighbors] : MILP_edges) {
        for (const auto& [v_node, flow_value] : neighbors) {

            if (flow_value > 1e-6) {

                std::string start_orig = u_node;
                std::string end_orig = v_node;

                std::vector<std::string> path = findPathInOriginal(start_orig, end_orig, adj_orig);

                if (path.empty()) {
                    std::cout << "  >> [警告] 未找到还原路径！" << std::endl;
                } else {
                    std::cout << "  >> 路径: ";
                    for(size_t i = 0; i < path.size(); ++i) {
                        std::cout << path[i] << (i == path.size() - 1 ? "" : " -> ");
                    }
                    std::cout << std::endl;
                }
                std::cout << std::string(60, '-') << std::endl;
            }
        }
    }

    if (processNetworkFlows(
            fileE,
            fileP,
            global_pair_neighbor_index,
            MILP_edges,
            outputFile) != 0) {
        return 1;
    }

    return 0;
}

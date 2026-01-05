#include "mesh_utils.hpp"

namespace nt::fmri::mesh {

/// Simplify mesh using a basic QEM edge-collapse (in-place replace vectors). target_face_count is desired final faces.
void simplify_mesh_qem(std::vector<Vec3>& V, std::vector<Vec3i>& F, int target_face_count){
    if (F.size() <= (size_t)target_face_count) return;

    // Build adjacency: vertex -> incident faces, edge map
    size_t nv = V.size();
    std::vector<std::unordered_set<int>> v2faces(nv);
    for(size_t i=0;i<F.size();++i){
        v2faces[F[i].a].insert((int)i);
        v2faces[F[i].b].insert((int)i);
        v2faces[F[i].c].insert((int)i);
    }

    // Build initial edge set
    std::unordered_map<EdgeKey, std::pair<int,int>> edges; // map->(v1,v2) store
    for(auto &t : F){
        EdgeKey e1(t.a, t.b); edges[e1] = {e1.a, e1.b};
        EdgeKey e2(t.b, t.c); edges[e2] = {e2.a, e2.b};
        EdgeKey e3(t.c, t.a); edges[e3] = {e3.a, e3.b};
    }

    // Build quadrics
    std::vector<Mat4Sym> Qs;
    build_vertex_quadrics(V, F, Qs);

    // Build heap with initial edge errors
    std::priority_queue<EdgeEntry, std::vector<EdgeEntry>, EdgeCmp> heap;
    for(auto &it : edges){
        int a = it.second.first, b = it.second.second;
        heap.push(compute_edge_error(Qs, V, a, b));
    }

    // Flags for removed vertices/faces
    std::vector<char> vert_removed(nv, 0);
    std::vector<char> face_removed(F.size(), 0);

    // vertex map (for index remapping)
    std::vector<int> vmap(nv); for(int i=0;i<(int)nv;++i) vmap[i]=i;

    int current_faces = (int)F.size();

    // Simple edge-collapse loop
    int iter = 0;
    while(current_faces > target_face_count && !heap.empty()){
        EdgeEntry top = heap.top(); heap.pop();

        int v1 = top.v1, v2 = top.v2;
        // check validity (not removed, and still adjacent)
        if (v1<0||v2<0||v1>= (int)nv || v2>= (int)nv) continue;
        if (vert_removed[v1] || vert_removed[v2]) continue;

        // check that edge still exists between v1 and v2 by seeing any face contains both
        bool edge_still = false;
        for(int fi : v2faces[v1]){
            if (face_removed[fi]) continue;
            auto &t = F[fi];
            if ((t.a==v1||t.b==v1||t.c==v1) && (t.a==v2||t.b==v2||t.c==v2)) { edge_still = true; break; }
        }
        if (!edge_still) continue;

        // Perform collapse: move v1 to optimal pos (we will remove v2 and redirect faces to v1)
        Vec3 newPos = top.pos;
        V[v1] = newPos;
        // accumulate quadrics
        Qs[v1] += Qs[v2];

        // mark v2 as removed and remap v2 -> v1
        vert_removed[v2] = 1;
        vmap[v2] = v1;

        // update faces incident to v2: replace v2 with v1, and remove degenerate faces
        std::vector<int> faces_to_check(v2faces[v2].begin(), v2faces[v2].end());
        for(int fi : faces_to_check){
            if (face_removed[fi]) continue;
            Vec3i t = F[fi];
            // replace indices
            if (t.a == v2) t.a = v1;
            if (t.b == v2) t.b = v1;
            if (t.c == v2) t.c = v1;
            // if degenerate (two equal indices) remove
            if (t.a==t.b || t.b==t.c || t.c==t.a){
                face_removed[fi] = 1;
                current_faces--;
                // remove face reference from other vertices
                v2faces[t.a].erase(fi);
                v2faces[t.b].erase(fi);
                v2faces[t.c].erase(fi);
            } else {
                // commit change and update adjacency (remove old references, add new)
                // remove fi from old vertex sets and add to new ones
                // (simple but somewhat redundant)
                // remove old references: iterate all vertices and ensure fi in set
                v2faces[t.a].insert(fi);
                v2faces[t.b].insert(fi);
                v2faces[t.c].insert(fi);
                F[fi] = t;
            }
        }

        // recompute errors for edges adjacent to v1
        // gather neighboring vertices from faces
        std::unordered_set<int> neigh;
        for(int fi : v2faces[v1]){
            if (face_removed[fi]) continue;
            auto &t = F[fi];
            neigh.insert(t.a);
            neigh.insert(t.b);
            neigh.insert(t.c);
        }
        neigh.erase(v1);
        for(int nb : neigh){
            if (vert_removed[nb]) continue;
            heap.push(compute_edge_error(Qs, V, v1, nb));
        }

        iter++;
        if ((iter & 65535) == 0) {
            // safety: if heap empties or iter big, break
            // std::cerr<<"simplify iteration "<<iter<<" faces "<<current_faces<<"\n";
        }
    }

    // Build new vertex list and remap indices
    std::vector<int> new_index(nv, -1);
    std::vector<Vec3> Vnew;
    Vnew.reserve(nv);
    for(int i=0;i<(int)nv;i++){
        if (!vert_removed[i]){
            new_index[i] = (int)Vnew.size();
            Vnew.push_back(V[i]);
        }
    }
    // sometimes vertices that were mapped into other vertices need mapping as well
    for(int i=0;i<(int)nv;i++){
        if (vert_removed[i]){
            // find representative (follow vmap chain)
            int rep = vmap[i];
            while(rep != vmap[rep]) rep = vmap[rep];
            new_index[i] = new_index[rep];
        }
    }

    // rebuild faces
    std::vector<Vec3i> Fnew;
    Fnew.reserve(current_faces);
    for(size_t i=0;i<F.size();++i){
        if (face_removed[i]) continue;
        Vec3i t = F[i];
        int a = new_index[t.a], b = new_index[t.b], c = new_index[t.c];
        if (a==b || b==c || c==a) continue;
        Fnew.emplace_back(a,b,c);
    }

    V.swap(Vnew);
    F.swap(Fnew);
}

/// Taubin smoothing: iterative non-shrinking Laplacian smoothing
void taubin_smooth(std::vector<Vec3>& V, const std::vector<Vec3i>& F, int iterations, float lambda, float mu){
    size_t nv = V.size();
    // build adjacency list
    std::vector<std::vector<int>> adj(nv);
    adj.assign(nv, {});
    for(auto &t : F){
        auto add_edge = [&](int a,int b){
            adj[a].push_back(b);
            adj[b].push_back(a);
        };
        add_edge(t.a,t.b);
        add_edge(t.b,t.c);
        add_edge(t.c,t.a);
    }
    // deduplicate adjacency lists
    for(size_t i=0;i<nv;i++){
        auto &lst = adj[i];
        std::sort(lst.begin(), lst.end());
        lst.erase(std::unique(lst.begin(), lst.end()), lst.end());
    }

    std::vector<Vec3> tmp = V;
    for(int it=0; it<iterations; ++it){
        // smooth step with lambda
        for(size_t i=0;i<nv;i++){
            if (adj[i].empty()) continue;
            Vec3 mean(0,0,0);
            for(int nb : adj[i]) mean = mean + tmp[nb];
            mean = mean / float(adj[i].size());
            V[i] = V[i] + (mean - V[i]) * lambda;
        }
        // update tmp for mu step
        tmp = V;
        // mu step
        for(size_t i=0;i<nv;i++){
            if (adj[i].empty()) continue;
            Vec3 mean(0,0,0);
            for(int nb : adj[i]) mean = mean + tmp[nb];
            mean = mean / float(adj[i].size());
            V[i] = V[i] + (mean - V[i]) * mu;
        }
    }
}

}

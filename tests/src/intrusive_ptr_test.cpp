#define NT_DEBUG_MODE
#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <unordered_set>
using namespace nt;

/*

intrusuve_ptr_test.cpp:
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -D__TBB_DYNAMIC_LOAD_ENABLED=0 -I/Users/sammoldenhauer/downloads/new_tensor/third_party/tbb/include -I/Users/sammoldenhauer/downloads/new_tensor/third_party/matplot/source -I/Users/sammoldenhauer/downloads/new_tensor/third_party/simde -I/Users/sammoldenhauer/downloads/new_tensor/third_party/half -I/Users/sammoldenhauer/downloads/new_tensor/third_party/stb -I/Users/sammoldenhauer/downloads/new_tensor/third_party/eigen -I/Users/sammoldenhauer/downloads/new_tensor/third_party/nifti_clib/znzlib -I/Users/sammoldenhauer/downloads/new_tensor/third_party/nifti_clib/niftilib -I/Users/sammoldenhauer/downloads/new_tensor/third_party/boost_config/include -I/Users/sammoldenhauer/downloads/new_tensor/third_party/multiprecision/include -I/Users/sammoldenhauer/downloads/new_tensor/third_party/uint128_t -I/Users/sammoldenhauer/downloads/new_tensor -I/Users/sammoldenhauer/downloads/new_tensor/test -DUSE_PARALLEL -O3 -DNDEBUG -std=c++17 -arch x86_64 -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.5.sdk -mmacosx-version-min=14.4 -march=native -fsanitize=address -g -MD -MT CMakeFiles/nt_tests.dir/tests/src/intrusive_ptr_test.cpp.o -MF CMakeFiles/nt_tests.dir/tests/src/intrusive_ptr_test.cpp.o.d -o CMakeFiles/nt_tests.dir/tests/src/intrusive_ptr_test.cpp.o -c /Users/sammoldenhauer/downloads/new_tensor/tests/src/intrusive_ptr_test.cpp

nt_tests:


/usr/local/Cellar/cmake/3.29.2/bin/cmake -E cmake_link_script CMakeFiles/nt_tests.dir/link.txt --verbose=1
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  -DUSE_PARALLEL -O3 -DNDEBUG -arch x86_64 -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.5.sdk -mmacosx-version-min=14.4 -Wl,-search_paths_first -Wl,-headerpad_max_install_names -fsanitize=address -g CMakeFiles/nt_tests.dir/tests/src/activation_func.cpp.o CMakeFiles/nt_tests.dir/tests/src/col_im_tests.cpp.o CMakeFiles/nt_tests.dir/tests/src/combinations_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/combine_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/compare_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/complex_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/conv_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/convert_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/dilate_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/dropout_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/fill_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/flip_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/index_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/intrusive_ptr_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/linear_autograd_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/main.cpp.o CMakeFiles/nt_tests.dir/tests/src/matmult_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/mesh_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/min_max_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/mutability_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/normalize_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/numpy_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/operator_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/padding_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/pooling_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/pytorch_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/repeat_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/round_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/save_load_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/softmax_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/sort_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/split_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/stride_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/sum_exp_log_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/tensor_grad_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/transpose_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/trig_test.cpp.o CMakeFiles/nt_tests.dir/tests/src/unique_test.cpp.o -o nt_tests  libneurotensor.a


*/


struct B; // forward declaration

struct A : public intrusive_ptr_target {
    intrusive_ptr<B> b; // Strong pointer to B
    static inline int destructor_calls = 0;
    ~A() {
        ++destructor_calls;
        // std::cout << "A destructor_calls\n";
    }
};

struct B : public intrusive_ptr_target {
    intrusive_ptr<A> a; // Strong pointer to A
    static inline int destructor_calls = 0;
    ~B() {
        ++destructor_calls;
        // std::cout << "B destructor_calls\n";
    }
};


struct B2;

struct A2 : public intrusive_ptr_target {
    weak_intrusive_ptr<B2> b; // Weak pointer to B
    static inline int destructor_calls = 0;
    A2()
    :b(intrusive_ptr<B2>(nullptr)) {}
    ~A2() {
        ++destructor_calls;
        // std::cout << "A2 destructor_calls\n";
    }
};


// the reslease_resources function is required for an intrusive_ptr
// because in an intrusive_ptr, the memory is not deleted until the weak_count reaches 0 as well
// this is different from a shared_ptr, this is required so there is no use after free error
// So, because A2 can house a weak_intrusive_ptr, B2 must delete "a" when the refcount is 0

struct B2 : public intrusive_ptr_target {
    intrusive_ptr<A2> a; // Strong pointer to A2
    static inline int destructor_calls = 0;
    void release_resources() override {
        a.reset();
    }
    ~B2() {
        ++destructor_calls;
        // std::cout << "B2 destructor_calls\n";
    }
};



// How to build a tree with intrusive ptr:
struct TreeNode : public intrusive_ptr_target {
    std::vector<intrusive_ptr<TreeNode>> children;
    weak_intrusive_ptr<TreeNode> parent;

    std::string name;

    static inline int destructor_calls = 0;

    TreeNode(std::string name) : parent(intrusive_ptr<TreeNode>(nullptr)), name(std::move(name)) {} 
    ~TreeNode() {
        ++destructor_calls;
        // std::cout << "Destroying: " << name << "\n";
    }

    void add_child(const intrusive_ptr<TreeNode>& child) {
        // A make-shift make_shared_from_this
        child->parent = weak_intrusive_ptr<TreeNode>(make_intrusive<TreeNode>(*this)); // weak assignment
        children.push_back(child);
    }

    void release_resources() override {
        children.clear(); // Clear strong refs to children
    }
};

struct MTreeNode {
    std::vector<intrusive_ptr<MTreeNode>> children;
    std::vector<weak_intrusive_ptr<MTreeNode>> parents;
    mutable std::atomic<int64_t> refcount_;
    mutable std::atomic<int64_t> weakcount_;

    std::string name;

    static inline int destructor_calls = 0;


    MTreeNode(std::string name) : name(std::move(name)) {} 
    ~MTreeNode() {
        ++destructor_calls;
    }

    intrusive_ptr<MTreeNode> intrusive_from_this() {
        intrusive_ptr<MTreeNode> out(this, detail::DontIncreaseRefCount{});
        detail::atomic_refcount_increment(refcount_);
        return std::move(out);
    }

    weak_intrusive_ptr<MTreeNode> weak_intrusive_from_this() {
        return weak_intrusive_ptr<MTreeNode>(intrusive_from_this());
    }
    
    void add_parent(const intrusive_ptr<MTreeNode>& child, const intrusive_ptr<MTreeNode>& p){
        p->children.push_back(child);
        parents.emplace_back(p);
    }
    
    template<typename... Args>
    void add_parent(const intrusive_ptr<MTreeNode>& child, const intrusive_ptr<MTreeNode>& p, Args&&... all_parents){
        p->children.push_back(child);
        parents.emplace_back(p);
        add_parent(child, std::forward<Args>(all_parents)...);
    }

    template<typename... Args>
    intrusive_ptr<MTreeNode> makeChild(std::string name, Args&&... parents){
        intrusive_ptr<MTreeNode> child = make_intrusive<MTreeNode>(std::move(name));
        child->add_parent(child, this->intrusive_from_this(), std::forward<Args>(parents)...);
        return std::move(child);
    }

    void release_resources(){
        children.clear(); // Clear strong refs to children
    }
};


struct IntrusiveString : public intrusive_ptr_target {
    std::string name;
    IntrusiveString(std::string n)
    :name(n){}
    IntrusiveString()
    :name("NULL") {}
};


// Now to create the computational graph:
struct GraphNode : public intrusive_ptr_target {
    std::vector<intrusive_ptr<GraphNode>> children;
    std::vector<weak_intrusive_ptr<GraphNode>> parents;
    intrusive_ptr<IntrusiveString> name;
    bool used;
    GraphNode()
    :name(make_intrusive<IntrusiveString>("NULL")), used(false) {}

    GraphNode(std::string n)
    :name(make_intrusive<IntrusiveString>(std::move(n))), used(false) {}

    void release_resources() override{
        children.clear();
    }

    void backward() {
        if(used) return;
        used = true;
        if(children.size() > 0){
            for(const auto& child : children){
                child->backward();
            }
            children.clear();
        }
        std::cout << "currently at before moving up: "<< name->name << std::endl;
        for(const auto& parent : parents){
            intrusive_ptr<GraphNode> node = parent.lock();
            if(node == nullptr)
                continue;
            node->backward();
        }
        parents.clear();
    }
};

struct Graph{
    mutable intrusive_ptr<GraphNode> node;
    Graph()
    :node(make_intrusive<GraphNode>()) {}

    Graph(std::string n)
    :node(make_intrusive<GraphNode>(std::move(n))) {}

    inline void add_parents(const Graph& child, const Graph& parent) const {
        child.node->parents.emplace_back(parent.node);
        parent.node->children.emplace_back(child.node);
    }

    template<typename... Args>
    inline void add_parents(const Graph& child, const Graph& parent, Args&&... parents) const {
        add_parents(child, parent);
        add_parents(child, std::forward<Args>(parents)...);
    }

    template<typename... Args>
    inline Graph ops_to_child(std::string child_name, Args&&... parents){
        Graph child(std::move(child_name));
        child.add_parents(child, *this, std::forward<Args>(parents)...);
        return child;
    }


    inline void backward(){
        node->backward();
    }
};

// Traversal function
void traverse_graph_impl(
    const intrusive_ptr<GraphNode>& node,
    std::vector<std::string>& visit_order,
    std::unordered_set<intrusive_ptr<GraphNode>>& visited
) {
    if (!node || visited.count(node)) return;
    visited.insert(node);

    // Traverse children
    for (const auto& child : node->children) {
        traverse_graph_impl(child, visit_order, visited);
    }

    // Visit current node after completing children
    visit_order.push_back(node->name->name);

    // traverse parents
    for (const auto& weak_parent : node->parents) {
        if (auto parent = weak_parent.lock()) {
            traverse_graph_impl(parent, visit_order, visited);
        }
    }
}

// Wrapper for ease of use
std::vector<std::string> traverse_graph(const Graph& start) {
    std::vector<std::string> result;
    std::unordered_set<intrusive_ptr<GraphNode>> visited;
    traverse_graph_impl(start.node, result, visited);
    return result;
}



void relu_track_func(Graph& graph){
    Graph child2_relu_where = graph.ops_to_child("child2_relu_where");
    Graph child2_relu_set = child2_relu_where.ops_to_child("child2_relu_set");
}


void intrusive_ptr_test(){
    run_test("intrusive ownership basic test", []{
        nt::intrusive_ptr<A> a = make_intrusive<A>();
        nt::utils::throw_exception(a.use_count() == 1, "use count is $", a.use_count());
        {
            nt::intrusive_ptr<A> b = a;
            nt::utils::throw_exception(a.use_count() == 2, "Error, use count should be 2 $", a.use_count());
        }
        nt::utils::throw_exception(a.use_count() == 1, "use count is $", a.use_count());
    });
    run_test("intrusive ownership weak test", []{
        nt::weak_intrusive_ptr<A> w(intrusive_ptr<A>(nullptr));
        {
            intrusive_ptr<A> a = make_intrusive<A>();
            w = weak_intrusive_ptr<A>(a);
            nt::utils::throw_exception(w.use_count() == 1, "Error, use count should be 1 $", w.use_count());
            nt::utils::throw_exception(w.lock(), "Error, should be able to lock weak intrusive ptr");
        }
        nt::utils::throw_exception(w.use_count() == 0, "Error, use count should be 0 $", w.use_count());
        nt::utils::throw_exception(!w.lock(), "Error, should not be able to lock"); // should be expired    
    });

    run_test("intrusive assignment reset test", []{
        intrusive_ptr<A> a = make_intrusive<A>();
        intrusive_ptr<A> b = make_intrusive<A>();
        a = b;
        nt::utils::throw_exception(a.use_count() == 2, "Error, use count should be 2 $", a.use_count());
    });
    run_test("intrusive self assignment test", []{
        intrusive_ptr<A> a = make_intrusive<A>();
        a = a;
        nt::utils::throw_exception(a.use_count() == 1, "Error, use count should be 1 $", a.use_count());
    });
    run_test("intrusive weak self assignment test", []{
        intrusive_ptr<A> _a = make_intrusive<A>();
        weak_intrusive_ptr<A> a(_a);
        a = a;
        nt::utils::throw_exception(a.weak_use_count() == 2, "Error, use count should be 2 $", a.weak_use_count());
    });


    // Successfully leaks
    // commented out
    // run_test("intrusive_ptr circular reference leaks", []{

    //     A::destructor_calls = 0;
    //     B::destructor_calls = 0;

    //     {
    //         intrusive_ptr<A> a__ = make_intrusive<A>();
    //         intrusive_ptr<B> b__ = make_intrusive<B>();

    //         a__->b = b__;
    //         b__->a = a__;
    //     }

    //     nt::utils::throw_exception(A::destructor_calls == 0, "Error: A should not be destructor_calls (leak expected)");
    //     nt::utils::throw_exception(B::destructor_calls == 0, "Error: B should not be destructor_calls (leak expected)");
    // });


    run_test("intrusive_ptr circular reference fixed with weak_intrusive_ptr", []{
        A2::destructor_calls = 0;
        B2::destructor_calls = 0;

        {
            auto a = make_intrusive<A2>();
            auto b = make_intrusive<B2>();
            a->b = weak_intrusive_ptr<B2>(b);  // weak
            b->a = a;  // strong

        }

        nt::utils::throw_exception(B2::destructor_calls == 1, "Error: B had $ destructor_calls", B2::destructor_calls);
        nt::utils::throw_exception(A2::destructor_calls == 1, "Error: A had $ destructor_calls", A2::destructor_calls);

    });

    run_test("intrusive_ptr one parent tree test", []{
        TreeNode::destructor_calls = 0;

        {
            auto root = make_intrusive<TreeNode>("root");
            auto child1 = make_intrusive<TreeNode>("child1");
            auto child2 = make_intrusive<TreeNode>("child2");

            root->add_child(child1);
            root->add_child(child2);

            auto grandchild = make_intrusive<TreeNode>("grandchild");
            child1->add_child(grandchild);
        }

        nt::utils::throw_exception(TreeNode::destructor_calls == 7, "Error: TreeNode had $ destructor_calls", TreeNode::destructor_calls);

    });

    run_test("intrusive_ptr multi parent tree test", []{
        MTreeNode::destructor_calls = 0;

        {
            intrusive_ptr<MTreeNode> root = make_intrusive<MTreeNode>("root");
            intrusive_ptr<MTreeNode> adult1 = make_intrusive<MTreeNode>("adult1");
            intrusive_ptr<MTreeNode> adult2 = make_intrusive<MTreeNode>("adult2");
            
            auto child1 = root->makeChild("child1", adult1);
            auto child2 = adult1->makeChild("child2", adult2);
            auto child3 = adult1->makeChild("child3", adult2, root);

            auto grandchild = child1->makeChild("grandchild", child2, child3);
        }
        std::cout << "Total destructors called: " << MTreeNode::destructor_calls << "\n";


    });

    run_test("intrusive graph test", []{
        Graph root("root");
        Graph adult1("adult1");
        Graph adult2("adult2");
        Graph adult3 = root.ops_to_child("adult3");

        Graph child1 = root.ops_to_child("child1", adult1);
        Graph child2 = root.ops_to_child("child2", adult1, adult3);
        Graph child3 = root.ops_to_child("child3", adult2, adult3);

        // Graph child2_relu_where = child2.ops_to_child("child2_relu_where");
        // Graph child2_relu_set = child2_relu_where.ops_to_child("child2_relu_set");
        relu_track_func(child2);

        Graph grandchild = adult1.ops_to_child("grandchild", adult2, adult3);
        // grandchild.backward();


        std::cout << "traversing..." << std::endl;
        std::vector<std::string> traversed = traverse_graph(grandchild);
        for(const auto& name : traversed)
             std::cout << name << " -> ";
        std::cout << "done" << std::endl;
        

    });

}

#ifndef NT_REFLECTION_DOCUMENTATION_REGISTRY_H__
#define NT_REFLECTION_DOCUMENTATION_REGISTRY_H__

#include <unordered_map>
#include <string>


namespace nt{
namespace doc{

using DocFunc = const char*();

struct DocRegistry {
    std::unordered_map<std::string, DocFunc*> entries;

    static DocRegistry& instance() {
        static DocRegistry registry;
        return registry;
    }

    void add(const std::string& name, DocFunc* doc_func) {
        entries[name] = doc_func;
    }

    void print(const std::string& name) {
        auto it = entries.find(name);
        if (it != entries.end()) {
            std::cout << it->second() << "\n";
        } else {
            std::cout << "No documentation found for: " << name << "\n";
        }
    }
};

#define NT_REGISTER_DOC(func_name, doc_string)                     \
    namespace {                                                 \
        const char* doc_##func_name() { return doc_string; }    \
        struct DocRegister_##func_name {                        \
            DocRegister_##func_name() {                         \
                nt::doc::DocRegistry::instance().add(#func_name, &doc_##func_name); \
            }                                                   \
        };                                                      \
        static DocRegister_##func_name _doc_register_##func_name; \
    }

}
}

#endif

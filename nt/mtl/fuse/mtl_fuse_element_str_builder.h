// This is part of the MTL fuse routine
// In GPU's you can at runtime make kernel functions
// In that same regard, may as well make kernel functions fuse together, so for example:
// nt::Tensor a = nt::rand(0, 10, device = nt::DeviceType::MTL);
// nt::Tensor b = nt::rand(0, 10, device = nt::DeviceType::MTL);
// nt::Tensor x = nt::rand(0, 10, device = nt::DeviceType::MTL);
// nt::Tensor c = nt::relu(a + b * x);
//
// now, in the above, a seperate kernel could be made for (a + b) and for that * x and then relu
// But, instead, it is just going to do it in one kernel

#include <cstring>
#include <vector>
#include <variant>
#include <functional>
#include <exception>

namespace nt::mtl::fuse{

// for example:
//  make_fuse("exp(", index<0>, ") + ", result_type<)
//  should result in this being added to the results vector:
//  results.emplace_back({
//      {FuseElementInfo::type::Expression, "exp("}, 
//      {FuseElementInfo::type::Index, 0},
//      {FuseElementInfo::type::Expression, ") + "},
//  });
struct FuseElementInfo {
    enum class type{
        Expression,
        Index,
        ResultType
    };
    // The result type is the prefix followed by a number
    // So if multiple fuse functions are combined, they would each have their own prefix for storing
    // just the index of the result, followed by a prefix
    
    std::string prefix = "A";
    size_t num_elements = 0;
    // a string expression, an indexed element, 
    // the result type represents {FuseElementHolders::index, results::index}
    using FuseElement = std::variant<std::string, size_t, std::pair<size_t, size_t>>;
    std::vector<std::vector<FuseElement>> results;

};

struct FuseElementHolders{
    std::vector<FuseElementInfo> infos;
    size_t last_result; // pointing to the index of the FuseElementInfo's that has the 
};

class FuseVars {
    public:
        template<size_t Var>
        struct Num {};

        template<size_t Var>
        struct Index {};

        enum class Functional{
            LastResult,
            SemiColon,
            Type
        };
};


inline void parse_arg(const FuseElementHolders& fused,
                std::vector<FuseElementInfo::FuseElement>& current, 
                const size_t& inputting_variables,
                const size_t& total_prev_variables,
                const std::string& text){
    // current += ' ';
    current.emplace_back(text);
}

inline void parse_arg(const std::vector<std::pair<int64_t, int64_t>>& conversions,
                std::vector<FuseElementInfo::FuseElement>& current, 
                const size_t& inputting_variables,
                const size_t& total_prev_variables,
                const std::string& text){
    // current += ' ';
    current.emplace_back(text);
}


inline std::string get_var_name(const size_t& inputting_variables, 
                        const size_t& total_prev_variables,
                        size_t Var){
    constexpr size_t min_a = size_t(char('A'));
    constexpr size_t max_a = size_t(char('Z'));
    constexpr size_t min_b = size_t(char('a'));
    constexpr size_t max_b = size_t(char('z'));
    static_assert(min_a < max_a && min_b < max_b, "Error with char interanl parsing");
    constexpr size_t dif = (max_a - min_a) + (max_b - min_b);
    size_t var_prev_zs = (Var > dif ? Var / dif : 0);
    size_t corrected_var = Var % dif;
    size_t total_prev_zs = (total_prev_variables > dif ? total_prev_variables / dif : 0);
    size_t corrected_total_prev = total_prev_variables % dif;
    total_prev_zs += var_prev_zs;
    size_t current_eval_num = corrected_var + corrected_total_prev;
    size_t add = (current_eval_num > dif ? 1 : 0);
    current_eval_num -= add;
    total_prev_zs += add;
    constexpr size_t half_dif = max_a - min_a;
    char n_char = char(current_eval_num > half_dif ? (current_eval_num - half_dif) + min_b : current_eval_num + min_a);
    // now put total_prev_zs before nchar
    std::string out(total_prev_zs + 1, 'z');
    out.back() = n_char;
    return std::move(out);
}


template<size_t Var>
inline std::string get_var_name(const size_t& inputting_variables, 
                        const size_t& total_prev_variables,
                        FuseVars::Num<Var>){
    constexpr size_t min_a = size_t(char('A'));
    constexpr size_t max_a = size_t(char('Z'));
    constexpr size_t min_b = size_t(char('a'));
    constexpr size_t max_b = size_t(char('z'));
    static_assert(min_a < max_a && min_b < max_b, "Error with char interanl parsing");
    constexpr size_t dif = (max_a - min_a) + (max_b - min_b);
    constexpr size_t var_prev_zs = (Var > dif ? Var / dif : 0);
    constexpr size_t corrected_var = Var % dif;
    size_t total_prev_zs = (total_prev_variables > dif ? total_prev_variables / dif : 0);
    size_t corrected_total_prev = total_prev_variables % dif;
    total_prev_zs += var_prev_zs;
    size_t current_eval_num = corrected_var + corrected_total_prev;
    size_t add = (current_eval_num > dif ? 1 : 0);
    current_eval_num -= add;
    total_prev_zs += add;
    constexpr size_t half_dif = max_a - min_a;
    char n_char = char(current_eval_num > half_dif ? (current_eval_num - half_dif) + min_b : current_eval_num + min_a);
    // now put total_prev_zs before nchar
    std::string out(total_prev_zs + 1, 'z');
    out.back() = n_char;
    return std::move(out);
}

template<size_t Var>
inline std::string get_indexed_var(const size_t& inputting_variables, 
                        const size_t& total_prev_variables,
                        FuseVars::Index<Var>){
    std::string var_name = get_var_name<Var>(inputting_variables, total_prev_variables, FuseVars::Num<Var>{});
    return var_name + ".data["+var_name+"_idx]";
}


template<size_t Var>
inline void parse_arg(const FuseElementHolders& fused,
                std::vector<FuseElementInfo::FuseElement>& current, 
                const size_t& inputting_variables,
                const size_t& total_variables,
                const FuseVars::Num<Var>& var){
    // current += ' ';'
    size_t var_num = total_variables + Var;
    current.emplace_back(var_num);
}

template<size_t Var>
inline void parse_arg(const std::vector<std::pair<int64_t, int64_t>>& conversions,
                std::vector<FuseElementInfo::FuseElement>& current, 
                const size_t& inputting_variables,
                const size_t& total_variables,
                const FuseVars::Num<Var>& var){
    // current += ' ';'
    size_t var_num = total_variables + Var;
    const std::pair<int64_t, int64_t>& p = conversions.at(Var);
    if(p.first == -1 && p.second == -1){
        current.emplace_back(var_num);
    }
    current.emplace_back(std::pair<size_t, size_t>(static_cast<size_t>(p.first), static_cast<size_t>(p.second)));
}

template<size_t Var>
inline void parse_arg(const FuseElementHolders& prev_fuse,
                        std::vector<FuseElementInfo::FuseElement>& current,
                        const size_t& inputting_variables,
                        const size_t& total_variables,
                        const FuseVars::Index<Var>& var){
    // current += ' ';
    size_t var_num = total_variables + Var;
    current.emplace_back(var_num);
    // current += get_indexed_var(inputting_variables, total_variables, var);
    // current += ' ';
}

template<size_t Var>
inline void parse_arg(const std::vector<std::pair<int64_t, int64_t>>& conversions,
                std::vector<FuseElementInfo::FuseElement>& current, 
                const size_t& inputting_variables,
                const size_t& total_variables,
                const FuseVars::Index<Var>& var){
    // current += ' ';'
    size_t var_num = total_variables + Var;
    const std::pair<int64_t, int64_t>& p = conversions.at(Var);
    if(p.first == -1 && p.second == -1){
        // std::cout << "Emplaicing back " << var_num << std::endl;
        current.emplace_back(var_num);
        return;
    }
    current.emplace_back(std::pair<size_t, size_t>(static_cast<size_t>(p.first), static_cast<size_t>(p.second)));
    // std::cout << "PAIREmplaceBack: " << std::get<2>(current.back()).first << std::endl;
}

inline void parse_arg(const FuseElementHolders& prev_fuse,
                        std::vector<FuseElementInfo::FuseElement>& current,
                const size_t& inputting_variables,
                const size_t& total_variables,
                const FuseVars::Functional& func){
    switch(func){
        case FuseVars::Functional::LastResult :
            current.emplace_back(std::pair<size_t, size_t>({prev_fuse.last_result, prev_fuse.infos[prev_fuse.last_result].results.size()-1}));
            return;
        case FuseVars::Functional::Type :
            current.emplace_back("{type}");
            // current += "{type}";
            return;
        default:
            return;
    }
}

inline void parse_arg(const std::vector<std::pair<int64_t, int64_t>>& conversions,
                        std::vector<FuseElementInfo::FuseElement>& current,
                const size_t& inputting_variables,
                const size_t& total_variables,
                const FuseVars::Functional& func){
    switch(func){
        case FuseVars::Functional::LastResult :
            throw std::logic_error("Last result used improperly");
            return;
        case FuseVars::Functional::Type :
            current.emplace_back("{type}");
            // current += "{type}";
            return;
        default:
            return;
    }
}

// so this is going to take a previous string
template<typename... Args>
inline FuseElementHolders build_fuse_element(const FuseElementHolders& fused, size_t inputting_variables, Args&&... args){
    std::vector<FuseElementInfo::FuseElement> elements;
    elements.reserve(sizeof...(Args));
    size_t total_prev_variables = fused.infos[fused.last_result].num_elements;
    ((parse_arg(fused, elements, inputting_variables, total_prev_variables, std::forward<Args>(args))), ...);
    FuseElementHolders out(fused);
    out.last_result = out.infos.size()-1;
    out.infos[out.last_result].results.emplace_back(std::move(elements));
    out.infos[fused.last_result].num_elements += inputting_variables;
    return std::move(out);
}


template<typename... Args>
inline FuseElementHolders build_fuse_element(size_t inputting_variables, Args&&... args){
    std::vector<FuseElementInfo::FuseElement> elements;
    elements.reserve(sizeof...(Args));
    size_t total_prev_variables = 0;
    FuseElementHolders out{
        std::vector<FuseElementInfo>{FuseElementInfo{"A", inputting_variables, std::vector<std::vector<FuseElementInfo::FuseElement>>() }},
        0
    };

    ((parse_arg(out, elements, inputting_variables, total_prev_variables, std::forward<Args>(args))), ...);
    out.infos[0].results.emplace_back(elements);
    return std::move(out);
}

// the first part of the pair refers to it's index, and then the holder
// inline std::string get_var_name(const size_t& inputting_variables, 
//                         const size_t& total_prev_variables,
//                         size_t Var){

template<typename... Args>
inline FuseElementHolders build_fuse_element(std::vector<std::pair<size_t, std::reference_wrapper<const FuseElementHolders>>> holders,
                                                size_t inputting_variables,
                                                Args&&... args){
    FuseElementHolders out{
        std::vector<FuseElementInfo>(),
        0
    };

    std::vector<size_t> sizes;
    sizes.reserve(holders.size());
    for(size_t i = 0; i < holders.size(); ++i){
        const auto& holder = holders[i];
        out.infos.insert(out.infos.end(), holder.second.get().infos.cbegin(), holder.second.get().infos.cend());
        sizes.push_back(i == 0 ? holder.second.get().infos.size() : holder.second.get().infos.size() + sizes[i-1]);
    }
    // size_t total = out.infos.size();
    // going to adjust all the prefixes and the result data:
    for(size_t i = 1; i < sizes.size(); ++i){
        for(size_t j = sizes[i-1]; j < sizes[i]; ++j){
            out.infos[j].prefix = get_var_name(sizes[i], 0, j);
            for(auto& results : out.infos[j].results){
                for(auto& result : results){
                    if(result.index() == 2){
                        std::get<2>(result).first += sizes[i-1];
                    }
                }
            }
        }
    }
    std::vector<std::pair<int64_t, int64_t>> conversions(inputting_variables, std::pair<int64_t, int64_t>({-1, -1}));
    for(size_t i = 0; i < holders.size(); ++i){
        conversions[holders[i].first] = std::pair<int64_t, int64_t>({holders[i].second.get().last_result + (i == 0 ? 0 : sizes[i-1]), 
                        holders[i].second.get().infos[holders[i].second.get().last_result].results.size()-1});
    }

    std::vector<FuseElementInfo::FuseElement> elements;
    elements.reserve(sizeof...(Args));
    size_t total_prev_variables = 0;
    ((parse_arg(conversions, elements, inputting_variables, total_prev_variables, std::forward<Args>(args))), ...);
    
    FuseElementInfo sub_info_{
        std::string(get_var_name(out.infos.size(), 1, 1)),
        inputting_variables - holders.size(),
        std::vector<std::vector<FuseElementInfo::FuseElement>>({std::move(elements)})
    };
    out.infos.emplace_back(std::move(sub_info_));
    out.last_result = out.infos.size()-1;
    return std::move(out);

}


inline std::string build_result(size_t index, const FuseElementInfo& info, const FuseElementHolders& holder){
    std::string out = "{type} " + info.prefix + get_var_name(info.results.size(), 0, index) + "_result = ";
    for(const auto& res_type : info.results[index]){
        if(res_type.index() == 0){
            // this is an expression
            out += std::get<0>(res_type);

        }else if(res_type.index() == 1){
            // this is an index
            size_t index = std::get<1>(res_type);
            std::string var_name = get_var_name(index, 0, index);
            out += info.prefix + var_name + "_val"; 
        }else if(res_type.index() == 2){
            // this is a result type
            std::pair<size_t, size_t> info_idx_res_idx = std::get<2>(res_type);
            out += holder.infos[info_idx_res_idx.first].prefix + get_var_name(info_idx_res_idx.second, 0, info_idx_res_idx.second) + "_result";
        }
    }
    return std::move(out);
}

inline std::string build_results(const FuseElementHolders& holders){
    std::string out = "";
    for(const auto& holder : holders.infos){
        for(size_t i = 0; i < holder.results.size(); ++i){
            out += "\t" + build_result(i, holder, holders) + ";\n";
        }
    }
    return std::move(out);
}

// template<typename... Args>
// inline std::string build_fuse_element(std::string_view prev_string, size_t inputting_variables, size_t total_prev_variables, Args&&... args){
//     std::string current = "";
//     ((parse_arg(prev_string, current, inputting_variables, total_prev_variables, std::forward<Args>(args))), ...);
//     return std::move(current);
// }

// this should really only be used internally by the fuse struct once being dispatched

namespace details{
inline void replace_all(std::string& str,
                 std::string_view from,
                 std::string_view to)
{
    if (from.empty()) return;

    std::string result;
    result.reserve(str.size());  // avoid repeated reallocations

    std::size_t pos = 0;
    std::size_t found;

    while ((found = str.find(from, pos)) != std::string::npos) {
        result.append(str, pos, found - pos);
        result += to;
        pos = found + from.size();
    }

    result.append(str, pos, std::string::npos);
    str.swap(result);
}
}


// it assumes all are constant
inline std::string build_fuse_element_function(const FuseElementHolders& fused_holders,
                                            std::string type,
                                            std::string name="fused_ops"){
    std::string start = "kernel void " + name + "(\n";
    size_t total_variables = 0;
    for(const auto& holder : fused_holders.infos){
        for(size_t i = 0; i < holder.num_elements; ++i)
            start += "\tconstant NtGPUTensor<" + type + ">& " + holder.prefix + get_var_name(holder.num_elements, 0, i) + " [[ buffer(" + std::to_string(i + total_variables) + ") ]],\n";
        total_variables += holder.num_elements;
    }
 
    start += "\tdevice " + type + "* out [[ buffer(" + std::to_string(total_variables) + ") ]],\n";
    start += "\tconstant uint3& grid_size [[ buffer(" + std::to_string(total_variables+1) + ") ]],\n";
    start += "\tconstant DispatchParams& dispatch_params_ [[ buffer(" + std::to_string(total_variables+2) + ") ]],\n";
    start += "\tuint3 tid [[thread_position_in_grid]]){\n";
    start += "\tulong idx = tid_to_gid(tid, grid_size) + dispatch_params_.start;\n";
    start += "\tif(idx >= dispatch_params_.end) return;\n\n";
    // compute indexes first
    for(const auto& holder : fused_holders.infos){
        for(size_t i = 0; i < holder.num_elements; ++i){
            std::string var_name = get_var_name(holder.num_elements, 0, i);
            start += "\tlong " + holder.prefix + var_name + "_idx = compute_offset(" + holder.prefix + var_name + ", idx);\n";
        }
    }
    start += "\n\n";
    // load data next
    for(const auto& holder : fused_holders.infos){
        for(size_t i = 0; i < holder.num_elements; ++i){
            std::string var_name = holder.prefix + get_var_name(holder.num_elements, 0, i);
            start += "\t{type} " + var_name + "_val = " + var_name + ".data[" + var_name + "_idx];\n";
        }
    }
    // do final computations and then store
    start += "\n\n";
    start += build_results(fused_holders);
    start += "\n\n";
    const auto& res_holder = fused_holders.infos[fused_holders.last_result];
    start += "\tout[idx] = " + res_holder.prefix + get_var_name(res_holder.results.size(), 0, res_holder.results.size()-1) + "_result;\n";
    details::replace_all(start, "{type}", type);
    start += "\n}";
    return std::move(start);
}

// // meant to be used in like a a += (fused function) type of way
// // In this case, the out function was already created, and there's no need to use the build_fuse_element
// inline std::string build_fuse_element_function_this_op(std::string_view prev_string,
//                                                         size_t total_variables,
//                                                         std::string type,
//                                                         std::string name = "fused_ops"){
//     std::string start = "kernel void " + name + "(\n";
//     for(size_t i = 0; i < total_variables-1; ++i){
//         start += "\tconstant NtGPUTensor<" + type + ">& " + get_var_name(total_variables, 0, i) + " [[ buffer(" + std::to_string(i) + ") ]],\n";
//     }
//     start += "\tNtGPUTensor<" + type + "> " + get_var_name(total_variables, 0, total_variables-1) + " [[ buffer(" + std::to_string(total_variables-1) + ") ]],\n";
//     start += "\tconstant uint3& grid_size [[ buffer(" + std::to_string(total_variables+1) + ") ]],\n";
//     start += "\tconstant DispatchParams& dispatch_params_ [[ buffer(" + std::to_string(total_variables+2) + ") ]],\n";
//     start += "\tuint3 tid [[thread_position_in_grid]]){\n";
//     start += "\tulong idx = tid_to_gid(tid, grid_size) + dispatch_params_.start;\n";
//     start += "\tif(idx >= dispatch_params_.end) return;\n";
//     for(size_t i = 0; i < total_variables; ++i){
//         std::string var_name = get_var_name(total_variables, 0, i);
//         start += "\tlong " + var_name + "_idx = compute_offset(" + var_name + ", idx);\n";
//     }
//     start += prev_string;
//     if(prev_string.back() != ';')
//         start += ';';
//     // start += "\tout[idx] = ";
//     // start += build_fuse_element(prev_string, total_variables, 0, FuseVars::Functional::LastResult, FuseVars::Functional::SemiColon);
//     details::replace_all(start, "{type}", type);
//     start += "\n}";
//     return std::move(start);
 
// }

}

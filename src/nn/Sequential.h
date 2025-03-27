#ifndef _NT_LAYERS_SEQUENTIAL_H_
#define _NT_LAYERS_SEQUENTIAL_H_

#include "../Tensor.h"
#include "Layer.h"
#include "Module.h"
#include "TensorGrad.h"
#include <type_traits>
#include <utility>
#include <vector>

namespace nt {
namespace layers {

class Sequential : public Module{
    std::vector<Layer> layers;
    std::vector<Layer> to_vec() { return {}; }
    template <typename T, typename... Args>
    inline std::vector<Layer> to_vec(T &&first, Args &&...rest) {
        static_assert(std::is_base_of_v<T, Module> || 
                      std::is_same_v<T, Layer> || 
                      std::is_same_v<T, std::vector<Layer> >, 
                      "Expected either a base of a Layer, a base of a Module, or a vector of layers");

        if constexpr (std::is_base_of_v<T, Module>){
            std::vector<Layer> result = {Layer(std::forward<T>(first))};
            if constexpr (sizeof...(Args) == 0){return std::move(result);}
            auto tail = to_vec(std::forward<Args>(rest)...);
            result.insert(result.end(), tail.begin(), tail.end());
            return result;
        }
        else if constexpr (std::is_same_v<T, Layer>){
            std::vector<Layer> result = {first};
            if constexpr (sizeof...(Args) == 0){return std::move(result);}
            auto tail = to_vec(std::forward<Args>(rest)...);
            result.insert(result.end(), tail.begin(), tail.end());
            return result;
        }
        else if constexpr (std::is_same_v<T, std::vector<Layer>>){
            if constexpr (sizeof...(Args) == 0){return std::move(first);}
            std::vector<Layer> result = std::forward<T>(first);
            auto tail = to_vec(std::forward<Args>(rest)...);
            result.insert(result.end(), tail.begin(), tail.end());
            return result;

        }
    }

    inline void init_layers(){
        int64_t counter = 0;
        for(auto& layer : layers){
            this->register_module("layer_["+std::to_string(++counter)+"]",layer); 
        }
    }

  public:
    template <typename T, typename... ls>
    Sequential(T&& first, ls&&... for_layer)
        : layers(to_vec(std::forward<T>(first), std::forward<ls>(for_layer)...)) {this->init_layers();}

    inline TensorGrad forward(TensorGrad x) {
        for(auto& layer : layers){
            x = layer(std::move(x));
        }
        return std::move(x);
    }

};

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Sequential, nt__layers__Sequential)


#endif

This is a breakdown of some of the features that are utility features specific to this package. Some of it is similar to that which is already in the standard C++ library, but with some key differences.

## Any Ref

`nt::utils::any_ref` is a class designed to hold rvalue and lvalue references. This contrasts to the `std::any` class that holds a copy of a value, made using `std::make_any`. The reason for this is to do something similar to what is below:

```C++
//example function that modifies a reference of a variable:
void modify_vector(std::vector<int>& vals){
    for(auto& val : vals){
        val += 10;
    }
}

//getting a moved variable
std::unique_ptr<int>&& modify_ptr(std::unique_ptr<int>&& ptr){
    *ptr = 30;
    return std::move(ptr);
}

//getting an value
int add_ten(int a){
    return a + 10;
}

int main(){
    std::vector<int> ints = {1, 2, 3, 4, 5, 6};
    std::unique_ptr<int> ptr_int = std::make_unique<int>();
    nt::utils::any_ref vec_any(ints); //this holds an lvalue reference of ints now
                                      //therefore it is important ints stays within the scope
    nt::utils::any_ref ptr_any(std::move(ptr_int));
    nt::utils::any_ref any_int(10);
    modify_vector(vec_any.cast<std::vector<int>&>());
    ptr_int = modify_ptr(ptr_int.cast<std::unique_ptr<int>&&>());
    int out_add_ten = add_ten(any_int.cast<int>());
    return 0;
}

```

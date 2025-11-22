file(REMOVE_RECURSE
  "libnt_functional_cpu.a"
  "libnt_functional_cpu.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/nt_functional_cpu.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

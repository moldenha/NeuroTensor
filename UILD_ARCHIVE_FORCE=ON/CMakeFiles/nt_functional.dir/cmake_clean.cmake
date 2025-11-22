file(REMOVE_RECURSE
  "libnt_functional.a"
  "libnt_functional.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/nt_functional.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

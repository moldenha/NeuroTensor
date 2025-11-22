file(REMOVE_RECURSE
  "libnt_core.a"
  "libnt_core.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/nt_core.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

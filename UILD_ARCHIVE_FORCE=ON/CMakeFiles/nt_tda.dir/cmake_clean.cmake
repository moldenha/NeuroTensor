file(REMOVE_RECURSE
  "libnt_tda.a"
  "libnt_tda.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/nt_tda.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

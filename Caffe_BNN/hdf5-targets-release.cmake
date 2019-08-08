#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hdf5-shared" for configuration "Release"
set_property(TARGET hdf5-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5-shared PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffehdf5.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5-shared "${_IMPORT_PREFIX}/lib/libcaffehdf5.lib" )

# Import target "hdf5-shared" for configuration "Release"
set_property(TARGET hdf5-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/caffehdf5.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/caffehdf5.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5-shared "${_IMPORT_PREFIX}/lib/caffehdf5.lib" "${_IMPORT_PREFIX}/bin/caffehdf5.dll" )

# Import target "hdf5_hl-shared" for configuration "Release"
set_property(TARGET hdf5_hl-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_hl-shared PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffehdf5_hl.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_hl-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_hl-shared "${_IMPORT_PREFIX}/lib/libcaffehdf5_hl.lib" )

# Import target "hdf5_hl-shared" for configuration "Release"
set_property(TARGET hdf5_hl-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_hl-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/caffehdf5_hl.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/caffehdf5_hl.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_hl-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_hl-shared "${_IMPORT_PREFIX}/lib/caffehdf5_hl.lib" "${_IMPORT_PREFIX}/bin/caffehdf5_hl.dll" )

# Import target "hdf5_cpp-shared" for configuration "Release"
set_property(TARGET hdf5_cpp-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_cpp-shared PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffehdf5_cpp.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_cpp-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_cpp-shared "${_IMPORT_PREFIX}/lib/libcaffehdf5_cpp.lib" )

# Import target "hdf5_cpp-shared" for configuration "Release"
set_property(TARGET hdf5_cpp-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_cpp-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/caffehdf5_cpp.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/caffehdf5_cpp.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_cpp-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_cpp-shared "${_IMPORT_PREFIX}/lib/caffehdf5_cpp.lib" "${_IMPORT_PREFIX}/bin/caffehdf5_cpp.dll" )

# Import target "hdf5_hl_cpp-shared" for configuration "Release"
set_property(TARGET hdf5_hl_cpp-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_hl_cpp-shared PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffehdf5_hl_cpp.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_hl_cpp-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_hl_cpp-shared "${_IMPORT_PREFIX}/lib/libcaffehdf5_hl_cpp.lib" )

# Import target "hdf5_hl_cpp-shared" for configuration "Release"
set_property(TARGET hdf5_hl_cpp-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf5_hl_cpp-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/caffehdf5_hl_cpp.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/caffehdf5_hl_cpp.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_hl_cpp-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_hl_cpp-shared "${_IMPORT_PREFIX}/lib/caffehdf5_hl_cpp.lib" "${_IMPORT_PREFIX}/bin/caffehdf5_hl_cpp.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

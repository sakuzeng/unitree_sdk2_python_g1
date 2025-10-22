#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "pcl_ros::pcl_ros_tf" for configuration "Release"
set_property(TARGET pcl_ros::pcl_ros_tf APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pcl_ros::pcl_ros_tf PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libpcl_ros_tf.so"
  IMPORTED_SONAME_RELEASE "libpcl_ros_tf.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS pcl_ros::pcl_ros_tf )
list(APPEND _IMPORT_CHECK_FILES_FOR_pcl_ros::pcl_ros_tf "${_IMPORT_PREFIX}/lib/libpcl_ros_tf.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

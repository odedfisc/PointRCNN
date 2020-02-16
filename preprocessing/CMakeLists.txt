project (InnovizFlow)
cmake_minimum_required(VERSION 3.5)

find_package( PythonInterp 3.6 REQUIRED )
find_package( PythonLibs 3.6 REQUIRED )

enable_testing()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
   find_package(PCL REQUIRED COMPONENTS common features filters segmentation registration io) # PCL::io causes linkage problem on Linux with Python bindings.
else()
   find_package(PCL REQUIRED COMPONENTS common features filters segmentation registration) 
endif()


option(INVZ_ENABLE_NETVIEW_VISUALIZATION "Enable Viewer Visualization" OFF)
if(INVZ_ENABLE_NETVIEW_VISUALIZATION)
	# Get protobuf from OpenCV
	find_package(OpenCV REQUIRED)
	include_directories(${Protobuf_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRs})
	include_directories(c:/Users/adis/Work/libs/opencv/3rdparty/protobuf/src)
	
	include_directories(invz_libs/viewer)	
	add_definitions(/D_NETVIEW_VISUALIZATION_ENABLED)
    add_subdirectory(invz_libs/viewer)

endif(INVZ_ENABLE_NETVIEW_VISUALIZATION)

include_directories(include)

if(MSVC)
   add_definitions(/D_CRT_SECURE_NO_WARNINGS)
   add_definitions(/D_SCL_SECURE_NO_WARNINGS)
   add_definitions(/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
   add_definitions(/DNOMINMAX)

#   add_definitions(/d2cgsummary /Bt+)
#   set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
#   set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
   
else()
   if(NOT CMAKE_BUILD_TYPE)
     set(CMAKE_BUILD_TYPE Release)
   endif()
   
   # on Linux use many warnings for correctness.
   set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra  -Wnon-virtual-dtor -pedantic") # -Wshadow
   set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wall -Wextra  -Wnon-virtual-dtor -pedantic") # -Wshadow
   
   
#   set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -fsanitize=leak -static-libasan -O -g")
#   set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined -fsanitize=leak -static-libasan -O -g")   
   
endif()

option(INVZ_ENABLE_PYTHON_BINDINGS "Build Python bindings" OFF)
if(INVZ_ENABLE_PYTHON_BINDINGS)
    add_subdirectory(external_libs/pybind11)
endif()

set(OPTIONAL_LITE_OPT_BUILD_TESTS OFF CACHE BOOL OFF FORCE)
# add_subdirectory(external_libs/span-lite)
# set(SPAN_LITE_OPT_BUILD_TESTS OFF CACHE BOOL OFF FORCE)
# add_subdirectory(external_libs/optional-lite)



add_subdirectory(modules/features_extractor)


option(INVZ_ENABLE_DRIVER_APPS "Build driver apps" ON)

if(INVZ_ENABLE_DRIVER_APPS)
	add_subdirectory(invz_libs/invzio)
	# add_subdirectory(apps/flow_driver)
	# add_subdirectory(apps/ego_motion_driver)
	add_subdirectory(apps/invz2_driver)
	# add_subdirectory(apps/ground_plane_driver)
	# add_subdirectory(apps/cluster_app)	
endif()

option(INVZ_ENABLE_PERF_TEST_APPS "Build perf-test apps" ON)
if(INVZ_ENABLE_PERF_TEST_APPS)
	# add_subdirectory(perf/ground_plane_alg_comparisons)
	add_subdirectory(perf/egomotion_sim_data_test)
	add_subdirectory(perf/ekf_sim_tests)	
endif()


option(INVZ_ENABLE_DOCTEST_UNIT_TESTS "Build doctest-based unit-tests" ON)
if(INVZ_ENABLE_DOCTEST_UNIT_TESTS)
	set(DOCTEST_WITH_TESTS OFF CACHE BOOL OFF FORCE)
	add_subdirectory(external_libs/doctest)
		
	add_executable(unit_tests modules/frame_keeper/frame_keeper_tests.cpp
							  modules/ground_plane/ground_plane_tests.cpp)
	target_link_libraries(unit_tests PUBLIC frame_keeper 
											ground_plane
											ego_motion
											doctest_with_main)
											
	if(INVZ_BUILD_DLO) 											
		target_link_libraries(unit_tests PUBLIC dlo)
		
		if(MSVC)
		   target_sources(unit_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/modules/ego_motion/dlo/dlo_tests.cpp)
		endif()
		
	endif()



    target_compile_definitions(unit_tests PRIVATE TEST_DATA_DIR=${PROJECT_SOURCE_DIR}/test_data)

	add_test(NAME unit_tests COMMAND unit_tests)

	set_property(TARGET doctest_with_main PROPERTY FOLDER "tests")
	set_property(TARGET unit_tests PROPERTY FOLDER "tests")
	
	
	if(INVZ_ENABLE_NETVIEW_VISUALIZATION)
		target_link_libraries(unit_tests PUBLIC net_view
													   ${OpenCV_LIBS}
													   libcurl)

		target_link_libraries(unit_tests PUBLIC debug     c:/Users/adis/Work/libs/opencv/build/vs2017.x64.dll/3rdparty/lib/Debug/libprotobufd.lib)
		target_link_libraries(unit_tests PUBLIC optimized c:/Users/adis/Work/libs/opencv/build/vs2017.x64.dll/3rdparty/lib/Release/libprotobuf.lib)
	endif()		

endif()






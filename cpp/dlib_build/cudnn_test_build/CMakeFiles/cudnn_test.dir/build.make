# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jiyao/nudo/dlib/dlib/cmake_utils/test_for_cudnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jiyao/nudo/cpp/dlib_build/cudnn_test_build

# Include any dependencies generated for this target.
include CMakeFiles/cudnn_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cudnn_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cudnn_test.dir/flags.make

CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o: CMakeFiles/cudnn_test.dir/flags.make
CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o: /home/jiyao/nudo/dlib/dlib/dnn/cudnn_dlibapi.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jiyao/nudo/cpp/dlib_build/cudnn_test_build/CMakeFiles $(CMAKE_PROGRESS_1)
	@echo "Building CXX object CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o -c /home/jiyao/nudo/dlib/dlib/dnn/cudnn_dlibapi.cpp

CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.i: cmake_force
	@echo "Preprocessing CXX source to CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jiyao/nudo/dlib/dlib/dnn/cudnn_dlibapi.cpp > CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.i

CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.s: cmake_force
	@echo "Compiling CXX source to assembly CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jiyao/nudo/dlib/dlib/dnn/cudnn_dlibapi.cpp -o CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.s

CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o.requires:
.PHONY : CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o.requires

CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o.provides: CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o.requires
	$(MAKE) -f CMakeFiles/cudnn_test.dir/build.make CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o.provides.build
.PHONY : CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o.provides

CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o.provides.build: CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o

# Object files for target cudnn_test
cudnn_test_OBJECTS = \
"CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o"

# External object files for target cudnn_test
cudnn_test_EXTERNAL_OBJECTS =

libcudnn_test.a: CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o
libcudnn_test.a: CMakeFiles/cudnn_test.dir/build.make
libcudnn_test.a: CMakeFiles/cudnn_test.dir/link.txt
	@echo "Linking CXX static library libcudnn_test.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cudnn_test.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cudnn_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cudnn_test.dir/build: libcudnn_test.a
.PHONY : CMakeFiles/cudnn_test.dir/build

CMakeFiles/cudnn_test.dir/requires: CMakeFiles/cudnn_test.dir/cudnn_dlibapi.cpp.o.requires
.PHONY : CMakeFiles/cudnn_test.dir/requires

CMakeFiles/cudnn_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cudnn_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cudnn_test.dir/clean

CMakeFiles/cudnn_test.dir/depend:
	cd /home/jiyao/nudo/cpp/dlib_build/cudnn_test_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiyao/nudo/dlib/dlib/cmake_utils/test_for_cudnn /home/jiyao/nudo/dlib/dlib/cmake_utils/test_for_cudnn /home/jiyao/nudo/cpp/dlib_build/cudnn_test_build /home/jiyao/nudo/cpp/dlib_build/cudnn_test_build /home/jiyao/nudo/cpp/dlib_build/cudnn_test_build/CMakeFiles/cudnn_test.dir/DependInfo.cmake
.PHONY : CMakeFiles/cudnn_test.dir/depend


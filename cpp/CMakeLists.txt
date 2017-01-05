cmake_minimum_required(VERSION 2.8)
project( detectFace )

include(../dlib/dlib/cmake)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")

find_package( OpenCV REQUIRED )

if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})

    ADD_EXECUTABLE(detectFace train_object_detector.cpp)
    TARGET_LINK_LIBRARIES(detectFace dlib ${OpenCV_LIBS} )

    ADD_EXECUTABLE(detectFaceFlip object_detector_flip.cpp)
    TARGET_LINK_LIBRARIES(detectFaceFlip dlib ${OpenCV_LIBS} )

else ()

    ADD_EXECUTABLE(detectFace train_object_detector.cpp)
    # Then we tell it to link with dlib.
    TARGET_LINK_LIBRARIES(detectFace dlib)

    ADD_EXECUTABLE(detectFaceFlip object_detector_flip.cpp)
    # Then we tell it to link with dlib.
    TARGET_LINK_LIBRARIES(detectFaceFlip dlib)
endif ()



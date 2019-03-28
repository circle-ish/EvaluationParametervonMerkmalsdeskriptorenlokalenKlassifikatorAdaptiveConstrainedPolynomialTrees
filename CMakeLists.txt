cmake_minimum_required(VERSION 2.8)


SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -ggdb -O0 -Wall -Wextra -DDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -ggdb -Wall -Wextra -DDEBUG")

SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -Wall -march=native")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

SET(CMAKE_BUILD_TYPE valgrind)
SET(CMAKE_C_FLAGS_VALGRIND "${CMAKE_C_FLAGS} -O3 -Wall")
SET(CMAKE_CXX_FLAGS_VALGRIND "${CMAKE_CXX_FLAGS} -Wall -O3")


SET(CMAKE_VERBOSE_MAKEFILE off)
set(SHOW_IMAGES 0)


INCLUDE(FindPkgConfig)
include(FindPackageHandleStandardArgs)

# --- general project setup ---

SET(NAME CPT)
PROJECT(${NAME})

SET(NAME_TRAINING CPT_Training)
SET(NAME_EVALUATION CPT_Evaluation)
SET(NAME_EVALUATION_LOCAL CPT_Evaluation_Local)
SET(NAME_BIN CPT_Track)

SET(VERSION_MAJOR "0")
SET(VERSION_MINOR "1")
SET(VERSION_REVISION "0")
SET(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION}")

SET(PREFIX ${CMAKE_INSTALL_PREFIX})
IF (NOT PREFIX)
SET(PREFIX $ENV{prefix})
ENDIF (NOT PREFIX)
message("PREFIX set to '${PREFIX}'")

SET(BINDIR "${PREFIX}/bin")
SET(INCLUDEDIR "${PREFIX}/include/face/${NAME}")
SET(MANDIR "${PREFIX}/man")
SET(LIBDIR "${PREFIX}/lib")
SET(SHAREDIR "${PREFIX}/share/face/")


SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#set(BUILD_EVAL ON CACHE BOOL "build evaluation executable")
#set(BUILD_TRAINING ON CACHE BOOL "build training executable")
#set(BUILD_TRACKING ON CACHE BOOL "build tracking executable")
set(BUILD_EVAL_LOCAL ON CACHE BOOL "build local evaluation executable")
        
#set(OpenCV_DIR "/vol/daimler/software/facialAnalysis/1.1-x86_64/share/OpenCV"
#   CACHE PATH "The path where OpenCVConfig.cmake is placed")



# --- general build ---
FIND_PACKAGE(OpenCV 2.4 REQUIRED COMPONENTS opencv_highgui opencv_core opencv_imgproc opencv_objdetect opencv_contrib)
FIND_PACKAGE(Boost 1.46 REQUIRED COMPONENTS program_options thread filesystem system regex)
#find_package(CUDA REQUIRED)

FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
        message("OPENMP FOUND")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS_VALGRIND "${CMAKE_CXX_FLAGS_VALGRIND} ${OpenMP_CXX_FLAGS}")

endif()


find_path(LOG4CXX_INCLUDE_DIR NAMES log4cxx/log4cxx.h "/usr/include")
find_library(LOG4CXX_LIBRARY NAMES log4cxx "/usr/lib")
# Set LOG4CXX_FOUND honoring the QUIET and REQUIRED arguments
find_package_handle_standard_args(LOG4CXX DEFAULT_MSG LOG4CXX_LIBRARY LOG4CXX_INCLUDE_DIR) 
     
if(LOG4CXX_FOUND)
     	MESSAGE("Found log4cxx")
ELSE (LOG4CXX_FOUND)
	MESSAGE("Error: Cannot find log4cxx")
ENDIF (LOG4CXX_FOUND)
  
  
LINK_DIRECTORIES(
        ${OpenCV_LIBRARY_DIRS} 
        ${Boost_LIBRARY_DIRS}
)  
		
INCLUDE_DIRECTORIES(
        ${OpenCV_INCLUDE_DIRS} 
        ${LOG4CXX_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
		../../bv/tclap/include
        src       
)

# --- Sources ---

SET (TRAINING_SRC
        src/main_training.cpp
        src/model/dataHandlers/DatasetHandler.cpp
        src/model/detectors/ViolaLienhartDetector.cpp
        src/model/local/Gauss.cpp
        src/model/local/LocalPolynoms.cpp
        src/model/local/PatchFeature.cpp
        src/model/local/Polynom.cpp
        src/model/global/GlobalTree.cpp
        src/model/trainers/Trainer.cpp
)

SET (EVALUATION_SRC
        src/main_evaluation.cpp
        src/model/dataHandlers/DatasetHandler.cpp
        src/model/detectors/ViolaLienhartDetector.cpp
        src/model/local/Gauss.cpp
        src/model/local/LocalPolynoms.cpp
        src/model/local/PatchFeature.cpp
        src/model/local/Polynom.cpp
        src/model/global/GlobalTree.cpp
        src/model/evaluation/Evaluator.cpp
        src/model/evaluation/TotalEvaluator.cpp
        src/model/trackers/CPT.cpp
)

SET (EVALUATION_LOCAL_SRC
        src/main_evaluation_local.cpp
        src/model/dataHandlers/DatasetHandler.cpp
        src/model/detectors/ViolaLienhartDetector.cpp
        src/model/evaluation/Evaluator.cpp
        src/model/evaluation/LocalEvaluator.cpp
        src/model/local/Gauss.cpp
        src/model/local/LocalPolynoms.cpp
        src/model/local/PatchFeature.cpp
        src/model/local/PatchFeatureUtils.cpp
        src/model/local/Polynom.cpp

)

SET (TRACKING_SRC
        src/main.cpp
        src/model/dataHandlers/DatasetHandler.cpp
        src/model/detectors/ViolaLienhartDetector.cpp
        src/model/local/Gauss.cpp
        src/model/local/LocalPolynoms.cpp
        src/model/local/PatchFeature.cpp
        src/model/local/Polynom.cpp
        src/model/global/GlobalTree.cpp
        src/model/trackers/CPT.cpp
)
                
# --- Targets ---
        
       
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_30)

                
IF (BUILD_TRAINING)
    ADD_EXECUTABLE(${NAME_TRAINING} 
        ${TRAINING_SRC}
    )
    TARGET_LINK_LIBRARIES(${NAME_TRAINING} 
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${LOG4CXX_LIBRARY}
    )   
    
    INSTALL(TARGETS ${NAME_TRAINING} DESTINATION ${BINDIR})
    
    SET(CONFIG_FILES config/configTraining.ini config/configTrainingLocal.ini)
    INSTALL(FILES ${CONFIG_FILES} DESTINATION ${SHAREDIR})
ENDIF(BUILD_TRAINING)

IF(BUILD_EVAL)
#    FIND_PACKAGE(Qt4 REQUIRED)
#    SET(helloworld_SOURCES src/view/mainwindow.cpp)
#    SET(helloworld_HEADERS src/view/mainwindow.h)
#    SET(helloworld_FORMS src/view/mainwindow.ui)
#    QT4_WRAP_CPP(helloworld_HEADERS_MOC ${helloworld_HEADERS})
#    QT4_WRAP_UI(helloworld_FORMS_HEADERS ${helloworld_FORMS})

#    INCLUDE(${QT_USE_FILE})
#    ADD_DEFINITIONS(${QT_DEFINITIONS})

    ADD_EXECUTABLE(${NAME_EVALUATION}
        ${EVALUATION_SRC}
#        ${helloworld_SOURCES}
#        ${helloworld_HEADERS_MOC}
#        ${helloworld_FORMS_HEADERS}
    )
        

    TARGET_LINK_LIBRARIES(${NAME_EVALUATION} 
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${LOG4CXX_LIBRARY}
#        ${QT_LIBRARIES}
    )     
        
    INSTALL(TARGETS ${NAME_EVALUATION} DESTINATION ${BINDIR})
    
    SET(CONFIG_FILES config/configEvaluation.ini)
    INSTALL(FILES ${CONFIG_FILES} DESTINATION ${SHAREDIR})

ENDIF(BUILD_EVAL)

IF(BUILD_EVAL_LOCAL)
    ADD_EXECUTABLE(${NAME_EVALUATION_LOCAL}
        ${EVALUATION_LOCAL_SRC}
    )


    TARGET_LINK_LIBRARIES(${NAME_EVALUATION_LOCAL}
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${LOG4CXX_LIBRARY}
    )

    INSTALL(TARGETS ${NAME_EVALUATION_LOCAL} DESTINATION ${BINDIR})

    SET(CONFIG_FILES config/configEvaluationLocal.ini)
    INSTALL(FILES ${CONFIG_FILES} DESTINATION ${SHAREDIR})

ENDIF(BUILD_EVAL_LOCAL)
        
    
IF(BUILD_TRACKING)
    ADD_EXECUTABLE(${NAME_BIN} 
        ${TRACKING_SRC}
    )
    TARGET_LINK_LIBRARIES(${NAME_BIN} 
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${LOG4CXX_LIBRARY}
    )           

   INSTALL(TARGETS ${NAME_BIN} DESTINATION ${BINDIR})

ENDIF(BUILD_TRACKING)

SET(CONFIG_FILES config/logger.cfg)
INSTALL(FILES ${CONFIG_FILES} DESTINATION ${SHAREDIR})

        
#CUDA_ADD_EXECUTABLE(CPT_CUDA_Track 
#        ${CPT_LIB_SOURCES}
#        src/mainCuda.cpp
#        src/cudaUtils/polyAnswerImage.cu
#        src/cudaUtils/helper.cpp
#)

#TARGET_LINK_LIBRARIES(CPT_CUDA_Track
#        ${OpenCV_LIBS}
#        ${Boost_LIBRARIES}
#        ${LOG4CXX_LIBRARY}
#)
        
# --- Install ---
        
#INSTALL(TARGETS ${NAME} LIBRARY DESTINATION ${LIBDIR})
#INSTALL(FILES ${CPT_INCLUDE_FILES} DESTINATION ${INCLUDEDIR})
#SET(PKGCONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.pc)
#CONFIGURE_FILE(pc.in ${PKGCONFIG_FILE})
#INSTALL(FILES ${PKGCONFIG_FILE} DESTINATION ${LIBDIR}/pkgconfig)

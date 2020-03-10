# build
file(GLOB IMGUI_SOURCE_FILES ${CMAKE_SOURCE_DIR}/3rdparty/imgui/*.cpp)
file(GLOB_RECURSE GUI_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE GUI_HEADER_FILES "*.h")

list(REMOVE_ITEM GUI_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/Native.h)
if (WIN32)
    list(REMOVE_ITEM GUI_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/NativeLinux.cpp)
elseif (APPLE)
    list(REMOVE_ITEM GUI_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/NativeWin32.cpp)
    list(REMOVE_ITEM GUI_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/NativeLinux.cpp)
    list(APPEND GUI_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/NativeMacOS.mm)
    list(APPEND GUI_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/MenuMacOS.mm)
else ()
    list(REMOVE_ITEM GUI_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/NativeWin32.cpp)
endif()

# create object library
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/3rdparty/imgui)
add_library(GUI OBJECT ${GUI_HEADER_FILES} ${GUI_SOURCE_FILES} ${IMGUI_SOURCE_FILES})
set_property(TARGET GUI PROPERTY CXX_STANDARD 14)
ShowAndAbortOnWarning(GUI)

# enforce 3rd party dependencies
add_dependencies(GUI build_all_3rd_party_libs)

# --- download resources -
list(APPEND TextureURLs "https://www.cgbookcase.com/textures/Wood/Wood_02/Wood_02_2K_AO.png")
list(APPEND TexturePaths "${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/wood_ao.png")
list(APPEND TextureURLs "https://www.cgbookcase.com/textures/Wood/Wood_02/Wood_02_2K_Base_Color.png")
list(APPEND TexturePaths "${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/wood_color.png")
list(APPEND TextureURLs "https://www.cgbookcase.com/textures/Wood/Wood_02/Wood_02_2K_Normal.png")
list(APPEND TexturePaths "${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/wood_normal.png")
list(APPEND TextureURLs "https://www.cgbookcase.com/textures/Wood/Wood_02/Wood_02_2K_Roughness.png")
list(APPEND TexturePaths "${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/wood_roughness.png")

list(APPEND TextureURLs "https://www.cgbookcase.com/textures/Metal/Bronze_01/Bronze_01_2K_Base_Color.png")
list(APPEND TexturePaths "${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/bronze_color.png")
list(APPEND TextureURLs "https://www.cgbookcase.com/textures/Metal/Bronze_01/Bronze_01_2K_Normal.png")
list(APPEND TexturePaths "${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/bronze_normal.png")
list(APPEND TextureURLs "https://www.cgbookcase.com/textures/Metal/Bronze_01/Bronze_01_2K_Roughness.png")
list(APPEND TexturePaths "${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/bronze_roughness.png")
list(APPEND TextureURLs "https://www.cgbookcase.com/textures/Metal/Bronze_01/Bronze_01_2K_Metallic.png")
list(APPEND TexturePaths "${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/bronze_metallic.png")

list(APPEND TextureURLs "https://www.cgbookcase.com/textures/Rock/Rock_01/Rock_01_2K_AO.png")
list(APPEND TexturePaths "${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/rock_ao.png")
list(APPEND TextureURLs "https://www.cgbookcase.com/textures/Rock/Rock_01/Rock_01_2K_Base_Color.png")
list(APPEND TexturePaths "${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/rock_color.png")
list(APPEND TextureURLs "https://www.cgbookcase.com/textures/Rock/Rock_01/Rock_01_2K_Normal.png")
list(APPEND TexturePaths "${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/rock_normal.png")
list(APPEND TextureURLs "https://www.cgbookcase.com/textures/Rock/Rock_01/Rock_01_2K_Roughness.png")
list(APPEND TexturePaths "${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/rock_roughness.png")

set(num_items 0)
list(LENGTH TextureURLs num_items)
math(EXPR num_items "${num_items} - 1")
foreach(index RANGE ${num_items})
    list(GET TextureURLs ${index} tURL)
    list(GET TexturePaths ${index} tPath)
    if (NOT EXISTS ${tPath})
        message(STATUS "downloading: ${tURL}")
        file(DOWNLOAD ${tURL} ${tPath} SHOW_PROGRESS STATUS DOWNLOAD_RESULT)
    else()
        message(STATUS "exist: ${tPath}")
    endif()
endforeach()

# --- build resources ----
file(GLOB GUI_RESOURCE_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/Resources/*.* ${CMAKE_CURRENT_SOURCE_DIR}/Resources/showcase/*)
file(GLOB GUI_MATERIAL_SOURCE_FILES "Materials/*.mat" "Materials/showcase/*.mat")

# copy GUI/Resources -> <output>/resources
set(GUI_RESOURCE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)
file(MAKE_DIRECTORY ${GUI_RESOURCE_DIR})
file(COPY ${GUI_RESOURCE_SOURCE_FILES}
     DESTINATION ${GUI_RESOURCE_DIR})

# Compile .mat files and copy to resources directory in output directory
if (IOS OR ANDROID)
    set(GUI_MATC_ARGS --platform mobile)
elseif (NOT WIN32)
    set(GUI_MATC_ARGS --platform desktop)
endif()
set(FILAMENT_MATC ${3RDPARTY_INSTALL_PREFIX}/bin/matc)

file(MAKE_DIRECTORY ${GUI_RESOURCE_DIR}/showcase)
set(GUI_MATERIAL_COMPILED_FILES "")
foreach(matsrc IN ITEMS ${GUI_MATERIAL_SOURCE_FILES})
    string(REGEX REPLACE "\\.[^.]*$" ".filamat" matdest ${matsrc})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/Materials/" "${GUI_RESOURCE_DIR}/" matdest ${matdest})
    list(APPEND GUI_MATERIAL_COMPILED_FILES ${matdest})
    add_custom_command(OUTPUT ${matdest}
                       COMMAND ${FILAMENT_MATC} ${GUI_MATC_ARGS} -o ${matdest} ${matsrc})
endforeach()

add_custom_target(GuiResources
                  DEPENDS ${GUI_MATERIAL_COMPILED_FILES})
add_dependencies(GuiResources filament_copy)
add_dependencies(GUI GuiResources)

# Export GUI_RESOURCE_FILES to parent CMake context (src/Open3D/)
set(GUI_RESOURCE_FILES
    ${GUI_RESOURCE_SOURCE_FILES} ${GUI_MATERIAL_COMPILED_FILES}
    PARENT_SCOPE)


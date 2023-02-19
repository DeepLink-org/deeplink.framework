FILE(GLOB _ATEN_SRCS
    *.cpp
    ops/*.cpp
    util/*.cpp)

LIST(APPEND ATEN_SRCS ${_ATEN_SRCS})

# Pass to parent
set(ATEN_SRCS ${ATEN_SRCS} PARENT_SCOPE)
# Run via:  cmake -DDEST=<path> -P cmake/DownloadMovieLens.cmake
#
# Downloads MovieLens 100K and extracts it under ${DEST}/ml-100k/.

if(NOT DEFINED DEST)
    message(FATAL_ERROR "DEST not set. Pass -DDEST=<output dir>.")
endif()

set(_url "https://files.grouplens.org/datasets/movielens/ml-100k.zip")
set(_zip "${DEST}/ml-100k.zip")
set(_extracted "${DEST}/ml-100k/u.data")

if(EXISTS "${_extracted}")
    message(STATUS "MovieLens 100K already present at ${DEST}/ml-100k/")
    return()
endif()

file(MAKE_DIRECTORY "${DEST}")

message(STATUS "Downloading ${_url}")
file(DOWNLOAD "${_url}" "${_zip}"
    SHOW_PROGRESS
    STATUS _dl_status
    TLS_VERIFY ON
)
list(GET _dl_status 0 _dl_code)
if(NOT _dl_code EQUAL 0)
    list(GET _dl_status 1 _dl_msg)
    message(FATAL_ERROR "Download failed: ${_dl_msg}")
endif()

message(STATUS "Extracting to ${DEST}")
file(ARCHIVE_EXTRACT INPUT "${_zip}" DESTINATION "${DEST}")
file(REMOVE "${_zip}")

if(NOT EXISTS "${_extracted}")
    message(FATAL_ERROR "Expected ${_extracted} after extraction, but it is missing.")
endif()

message(STATUS "Done. u.data is at ${_extracted}")

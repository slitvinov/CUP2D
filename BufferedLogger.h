//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#ifndef CubismUP_3D_utils_BufferedLogger_h
#define CubismUP_3D_utils_BufferedLogger_h

#include <unordered_map>
#include <sstream>

/*
 * Buffered file logging with automatic flush.
 *
 * A stream is flushed periodically.
 * (Such that the user doesn't have to manually call flush.)
 *
 * If killing intentionally simulation, don't forget to flush the logger!
 */
class BufferedLogger {
    struct Stream {
        std::stringstream stream;
        int requests_since_last_flush = 0;
        // GN: otherwise icpc complains
        Stream(const Stream& c) {}
        Stream() {}
    };
    typedef std::unordered_map<std::string, Stream> container_type;
    container_type files;

    /*
     * Flush a single stream and reset the counter.
     */
    void flush(container_type::iterator it);
public:

    ~BufferedLogger() {
        flush();
    }

    /*
     * Get or create a string for a given file name.
     *
     * The stream is automatically flushed if accessed
     * many times since last flush.
     */
    std::stringstream& get_stream(const std::string &filename);

    /*
     * Flush all streams.
     */
    inline void flush(void) {
        for (auto it = files.begin(); it != files.end(); ++it)
            flush(it);
    }
};

extern BufferedLogger logger;  // Declared in BufferedLogger.cpp.

#endif

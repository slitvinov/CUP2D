BEGIN {

    for (i = 1; i in ARGV; i++) {
	f(ARGV[i])
    }

}

function f(path,   include) {

    while (getline < path == 1) {

	if (/^#include "/) {
	    include = path
	    sub(/^#include "/, "")
	    sub(/"[ \t]*/, "")
	    if (!($0 in a)) {
		a[$0]
		print $0 | "cat >&2"
		f($0)
	    }
	} else {
	    print($0)
	}
    }
}

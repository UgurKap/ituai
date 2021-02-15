# Uğur Ali Kaplan
# 150170042

from main import *

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc < 2:
        print("No input file")
        exit(0)

    in_file = sys.argv[1]
    out_file = "output.txt"
    if argc >= 3:
        out_file = sys.argv[2]

    sys.stdout = open(out_file, "w")
    w = World(in_file)
    w.bfs()
    del w

    sys.stdout.close()
import re
import sys,os

CONTENT = 'export IS_cover=False\n'

def C_coverage(coveragedir, require_coverage):
    c_coverage_file = coveragedir + "increment.txt"
    with open(c_coverage_file, 'r') as file:
        tracefile = file.read()

    lines = tracefile.split("\n")

    for line in lines:
        coverage_percent = re.search(r'\|(.+?)%', line)
        if coverage_percent and float(coverage_percent.group(1)) < float(require_coverage):
            with open(os.path.join(coveragedir, 'IS_cover.txt'), 'a') as file:
                file.write(CONTENT)

if __name__ == '__main__':
    coveragedir, require_coverage = sys.argv[1:3]
    if os.path.exists(os.path.join(coveragedir, 'increment.info')):
        C_coverage(coveragedir, require_coverage)

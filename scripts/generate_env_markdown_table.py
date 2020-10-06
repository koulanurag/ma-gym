"""
Usage: python scripts/generate_env_markdown_table.py
"""

import argparse
from os import listdir
from os.path import join

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Markdown table for ReadMe')
    parser.add_argument('--path', default='static/gif/',
                        help='Path (default: %(default)s)')
    args = parser.parse_args()

    onlyfiles = [(f, join(args.path, f)) for f in sorted(listdir(args.path))]

    for i in range(0, len(onlyfiles), 3):

        msg = "|"
        for f, file_path in onlyfiles[i:i + 3]:
            msg += ' __' + f.split('.')[0] + '__ |'
        if i == 0:
            msg += '\n'
            msg += '|' + ''.join(':---:|' for _ in range(3))
        print(msg)

        msg = "|"
        for f, file_path in onlyfiles[i:i + 3]:
            msg += '![' + f + '](' + file_path + ')|'
        print(msg)

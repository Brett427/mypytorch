# -*- coding:utf8 -*-
# author: Brett

# 基本用法 ×××××××××××××××××××××××××××××××××××××××××××××××××××××××
import argparse
parser = argparse.ArgumentParser()
parser.parse_args()

# $ python3 usage.py

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("echo")
args = parser.parse_args()
print(args.echo)
# add_argument  表示我们说明了命令行需要接受哪些参数。
# 多个参数时按顺序直接写在命令行后面。不用加具体的名字。但是可选参数的话就要加。
# $ python3 prog.py
# usage: prog.py [-h] echo
# prog.py: error: the following arguments are required: echo

# $ python3 prog.py --help
# usage: prog.py [-h] echo

# positional arguments:
#   echo

# optional arguments:
#   -h, --help  show this help message and exit

# $ python3 prog.py foo
# foo

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("echo", help="echo the string you use here")
args = parser.parse_args()
print(args.echo)
# $ python3 prog.py -h
# usage: prog.py [-h] echo
#
# positional arguments:
#   echo        echo the string you use here
#
# optional arguments:
#   -h, --help  show this help message and exit

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("square", help="display a square of a given number",
                    type=int)
args = parser.parse_args()
print(args.square**2)
# 必须声明类型，默认是字符串。
# $ python3 prog.py 4
# 16
# $ python3 prog.py four
# usage: prog.py [-h] square
# prog.py: error: argument square: invalid int value: 'four'

# 高级用法××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
#  1可选参数(可以选择不加)
#  -- 表示是可选参数。 -也是。
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ver", help="increase output verbosity")
args = parser.parse_args()
if args.ver:
    print("verbosity turned on")
    print(args.ver)

# $ python3 prog.py

# $ python3 prog.py --ver 1
# verbosity turned on
# 1

# $ python3 prog.py --help
# usage: prog.py [-h] [--verbosity VERBOSITY]
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --verbosity VERBOSITY
#                         increase output verbosity

# 默认可选参数会被赋值为str，而在实际情况中，很多时候，可选参数是作为一个标识而不是一个确切的值，
# 仅需要确定是true或false即可，可以指定关键字action，赋值为"store_true"：
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-ver", help="increase output verbosity",action="store_true")
args = parser.parse_args()
if args.ver:
    print("dsds")
# 加了action="store_true"以后,就不用输入具体值了。
# Python prog.py -ver 就会输出 dsds.



# 2短命令，后面是别名。
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity",)
args = parser.parse_args()
if args.verbose:
    print("verbosity turned on")

